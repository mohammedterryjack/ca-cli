#!/usr/bin/env python3

import click
from pathlib import Path
import numpy as np
from pyfiglet import Figlet
from tempfile import TemporaryDirectory
from shutil import copy 

from tools.utils import (
    FFNN_ONESHOT, 
    FFNN_ITERATIVE, 
    Activation, 
    render_halfblock, 
    combine_side_by_side, 
    save_evolution, 
    first_differing_row,
    cosine_similarity_between_evolutions,
    bar_plot,
    load_evolutions_from_parquet,
    save_evolutions_as_parquet,
    train,
    load_model,
    generate_text,
    EOL,
    BOS,
    SPECIAL, 
    evolution_to_parquet_string,
    evolution_and_emergence_to_parquet_string,
    safe_string_to_evolution,
    save_evolutions_as_parquet,
    line_plot 
)

DEFAULT_HIDDEN_LAYERS = [50, 100, 50]
DEFAULT_ACTIVATION = "relu"
DEFAULT_BINARISATION = 0.5
DEFAULT_LEARN = "iterative"

@click.command(name="learn")
@click.option("--input", "-i", type=Path, required=True,
              help="Input Parquet file")
@click.option("--additional-input-features", "-ii", type=Path, default=None,
              help="Parquet file containing additional features.")
@click.option("--output", "-o", type=Path, default=None,
              help="Output directory for weights and predictions.")
@click.option("--learn", "-l",
              type=click.Choice(["iterative", 'one-shot', "generative"]),
              default=DEFAULT_LEARN, help="Iterative = FFNN with Backprop. One-shot = FFNN with Pseudo-Inverse. Generative = char-GPT")
@click.option("--weights", "-w", type=Path, default=None,
              help="Optional: load pretrained weights instead of training.")
@click.option("--hidden-layers", "-h", type=int, multiple=True,
              default=DEFAULT_HIDDEN_LAYERS)
@click.option("--activation", "-a",
              type=click.Choice(["relu", "tan"]),
              default=DEFAULT_ACTIVATION)
@click.option("--binarisation-threshold", "-b", type=float,
              default=DEFAULT_BINARISATION)
@click.option("--max-iterations", "-n", type=int, default=2000)
@click.option("--learning-rate", "-lr", type=float, default=1e-3)
@click.option("--heads", type=int, default=4)
@click.option("--embedding-dimension", "-d", type=int, default=256)
def learn_cli(
    input:Path, 
    additional_input_features:Path|None, 
    output:Path|None, 
    learn:str, 
    weights:Path|None, 
    hidden_layers:list[int], 
    activation:str, 
    binarisation_threshold:float,
    learning_rate:float,
    max_iterations:int,
    heads:int,
    embedding_dimension:int
) -> None:
    predicted_evolutions_to_save = {}

    banner = Figlet(font="smslant").renderText("Learning")
    click.secho(banner, fg="red", bold=True)

    if not input.is_file() or input.suffix != ".parquet":
        raise click.ClickException(f"Input must be a Parquet file, got {input}")

    # -------------------------
    # Determine output directory
    # -------------------------
    if output is None:
        output = input.parent / f"{input.stem}_learn"
    output.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output directory {output}")

    suffix_map = {"one-shot": ".npz", "iterative": ".joblib", "generative":""}
    weight_file = output / f"weights{suffix_map[learn]}"


    # -------------------------
    # Additional Input Features
    # -------------------------

    loaded_evolutions = load_evolutions_from_parquet(input)

    if additional_input_features:
        if not additional_input_features.is_file() or additional_input_features.suffix != ".parquet":
            raise click.ClickException(f"Additional Input must be a Parquet file, got {additional_input_features}")
        loaded_additional_evolutions = load_evolutions_from_parquet(additional_input_features)
    else:
        loaded_additional_evolutions = None

    # -------------------------
    # Load all evolutions �~F~R build big dataset
    # -------------------------
    X_list = []
    Y_list = []
    evo_per_image = {}   # store per-image evolutions for later prediction

    for name,evo in loaded_evolutions.items():
        X = evo[:-1]
        Y = evo[1:]
 
        if loaded_additional_evolutions is not None:
            if name not in loaded_additional_evolutions:
                raise click.ClickException(f"{name} Not found in additional features")
            additional_features = loaded_additional_evolutions[name]
            X_extra = additional_features[:-1]
            if len(X_extra) != len(X):
                raise click.ClickException(
                    f"Shape mismatch for {name} and additional features "
                )
            X = np.hstack([X, X_extra])

        X_list.append(X)
        Y_list.append(Y)
        evo_per_image[name] = (X, Y)

        #click.echo(f"Loaded {name} �~F~R {evo.shape}")

    X_full = np.vstack(X_list)
    Y_full = np.vstack(Y_list)

    click.echo(f"\nTraining dataset shape: X={X_full.shape}, Y={Y_full.shape}")


    if learn=='generative':
        if weights is None:
            with TemporaryDirectory() as temp_dir:
                path_temp_dir = Path(temp_dir)
                vocab_str = ''.join(map(str, np.unique(X_full))) + EOL
                if additional_input_features is None:  
                    copy(input, path_temp_dir / input.name)
                    copy(input, path_temp_dir / f"2_{input.name}")
                    vocab_str = ''.join(map(str, np.unique(X_full))) + EOL 
                else:
                    save_evolutions_as_parquet(
                        evolutions=loaded_evolutions,
                        additional_features=loaded_additional_evolutions,
                        output_dir=path_temp_dir / input.stem 
                    )
                    copy(path_temp_dir / input.name, path_temp_dir / f"2_{input.name}") 
                    vocab_str += ''.join(SPECIAL.keys())
                loss = train(
                    parquet_dir=path_temp_dir,
                    vocab_string=vocab_str,
                    batch_size=1,
                    lr=learning_rate,
                    steps=max_iterations,
                    out_dir=weight_file,
                    resume_checkpoint=None,
                    sequence_len=1024,
                    layers=len(hidden_layers),
                    heads=heads,
                    embedding_dimension=embedding_dimension
                )
            loss_curve = line_plot(loss)
            click.secho(loss_curve, fg="green")
        else:
            weight_file = weights 
        model, vocab = load_model(weight_file, "final.pt")
 
    else:

        # -------------------------
        # Activation enum
        # -------------------------
        activation_map = {"relu": Activation.RELU, "tan": Activation.TAN}
        activation_enum = activation_map[activation]

        # -------------------------
        # Init model
        # -------------------------
        FFNN = FFNN_ONESHOT if learn == "one-shot" else FFNN_ITERATIVE
        input_dim = X_full.shape[1]
        output_dim = Y_full.shape[1]

        model = FFNN(
            input_dimension=input_dim,
            hidden_dimensions=list(hidden_layers),
            output_dimension=output_dim,
            activation=activation_enum,
            learning_rate=learning_rate,
            max_iterations=max_iterations
        )

        # -------------------------
        # Train or load weights
        # -------------------------
        if weights is not None:
            model.load(weights)
            click.secho(f"Loaded pre-trained weights from {weights}", fg="green")
        else:
            click.echo("Training model on ALL images...")
            model.fit(X_full, Y_full)
            model.save(weight_file)
            click.secho(f"Saved model weights → {weight_file}", fg="green")

    # -------------------------
    # Prediction directories
    # -------------------------
    pred_raw_dir = output / "predictions"
    pred_bin_dir = output / "predictions_binarised"
    pred_raw_dir.mkdir(exist_ok=True)
    pred_bin_dir.mkdir(exist_ok=True)

    # -------------------------
    # Predict for each image separately
    # -------------------------
    for name, (X_img, Y_img) in evo_per_image.items():
        click.echo(f"\nPredicting for {name}")

        if learn == "generative":
            X_img_str = evolution_to_parquet_string(X_img) 
            #if additional_input_features is None: 
            #    X_img_str = evolution_to_parquet_string(X_img)
            #else:
                #X_emergence = loaded_additional_evolutions[name] 
                #X_img_str = evolution_and_emergence_to_parquet_string(evolution=X_img, emergence_mask=X_emergence) 
            prompt = X_img_str.split(EOL)[0] + EOL 
            click.echo(prompt) 
            Y_raw = generate_text(
                model=model,
                vocab=vocab,
                prompt=prompt,
                length=model.config.sequence_len,
                temperature=1.0,
                top_k=None,
            )
            try: 
                Y_hat = safe_string_to_evolution(Y_raw,decode_special=additional_input_features is not None)
                if not any(Y_hat):
                    raise Exception("No Y_hat")
                Y_hat = Y_hat[1:]
            except Exception as e:
                click.echo(f"{e}:Y_hat: {Y_raw.replace(EOL,'\n')}")
                Y_hat = None
        else: 
            Y_raw = model.predict(X_img)
            Y_hat = (Y_raw > binarisation_threshold).astype(int)

        if Y_hat is None:
             continue 
       
        try:
            Y_img_ = Y_img[:len(Y_hat)] 

            diff = first_differing_row(values1=Y_img_, values2=Y_hat)
            click.echo(f"First differing row: {diff}")

            left = render_halfblock(values=Y_img_)
            right = render_halfblock(values=Y_hat)
            click.echo(combine_side_by_side(left, right))

            similarities = cosine_similarity_between_evolutions(a=Y_hat, b=Y_img_)
            bar_plot(similarities)
       
            #pred_raw = pred_raw_dir / f"{name}.png"
            pred_bin = pred_bin_dir / f"{name}.png"
            #save_evolution(Y_raw, pred_raw)
            save_evolution(Y_hat, pred_bin)
            #click.secho(f"Saved raw prediction {pred_raw}", fg="green")
            click.secho(f"Saved bin prediction {pred_bin}", fg="green")
 
        except Exception as e:
            click.echo(f"{e}: Y_hat: {Y_raw.replace(EOL,'\n')}  {Y_hat}") 
        predicted_evolutions_to_save[name] = Y_hat

    save_evolutions_as_parquet(evolutions=predicted_evolutions_to_save, output_dir=output)
    click.secho(f"Saved predictions to {output}.parquet", fg='green')

if __name__ == "__main__":
    learn_cli()

