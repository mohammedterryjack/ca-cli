from pathlib import Path 
from typing import List,Generator,Optional
from math import e,pi,log
from scipy.stats import linregress
from statistics import median

from numpy import cumsum
from matplotlib.pyplot import plot, savefig, clf, figure, xlabel,ylabel

class O1TestForChaos:
    @staticmethod
    def _transform(observables:List[float],angle:float) -> Generator[complex,None,None]:
        for t,phi_t in enumerate(observables):
            yield phi_t * e**(1j*t*angle)

    @staticmethod
    def transform(observables:List[float],angle:float,path_save:Path|None = None) -> List[complex]:
        z = cumsum(list(O1TestForChaos._transform(observables=observables,angle=angle)))
        if path_save:
            clf()
            figure() 
            plot(z.real,z.imag)
            xlabel("Real(z)")
            ylabel("Imag(z)")
            savefig(path_save / "transform.png")
        return z.tolist()

    @staticmethod
    def _mean_square_displacement_term(a:complex) -> float:
        return a.real **2 + a.imag **2

    @staticmethod
    def _mean_square_displacement(transformed_data:List[complex], N:int) -> Generator[float,None,None]:
        T = len(transformed_data)
        for n in range(1,N):        
            m = sum(map(
                lambda t:O1TestForChaos._mean_square_displacement_term(
                    a = transformed_data[n+t] - transformed_data[t]
                ), 
                range(T-N)
            ))
            yield m/(T-N)

    @staticmethod
    def mean_square_displacement(transformed_data:List[complex], N:int, path_save:Path|None = None) -> List[float]:
        M = list(O1TestForChaos._mean_square_displacement(transformed_data=transformed_data,N=N)) 
        if path_save:
            clf()
            figure() 
            plot(M)
            xlabel("n")
            ylabel("M")
            savefig(path_save / "mean_square_displacement.png")
        return M

    @staticmethod
    def correlation_coefficient(mean_square_displacement:List[float], N:int, path_save:Path|None = None) -> float:
        log_M = list(map(lambda M_n:log(M_n) if M_n else 0.0,mean_square_displacement))
        log_N = list(map(log,range(1,N)))
        slope, intercept, K, _, _ = linregress(log_N, log_M)
        if path_save:
            clf()
            figure() 
            best_fit = list(map(lambda value:slope*value+intercept, log_N))
            plot(log_N,log_M)
            plot(log_N, best_fit)
            xlabel("log N")
            ylabel("log M")
            savefig(path_save / "correlation_coefficient.png")
        return K

    @staticmethod
    def _test_for_chaos(observables:List[float],n_angles:int,N:int,path_save:Path|None = None) -> Generator[float,None,None]:
        for n in range(1,n_angles+1):
            c = (n/20)*pi
            z = O1TestForChaos.transform(observables=observables, angle=c, path_save=path_save if n==1 else None)
            M = O1TestForChaos.mean_square_displacement(transformed_data=z, N=N,path_save=path_save if n==1 else None) 
            K = O1TestForChaos.correlation_coefficient(mean_square_displacement=M, N=N, path_save=path_save if n==1 else None)
            yield K

    @staticmethod
    def test_for_chaos(observables:List[float],n_angles:int,N:Optional[int]=None,path_save:Path|None = None) -> float:
        Ks = list(O1TestForChaos._test_for_chaos(
            observables=observables,
            n_angles=n_angles,
            N=len(observables)//3 if N is None else N,
            path_save=path_save
        ))
        if path_save:
            clf()
            figure() 
            angles = list(map(lambda n:(n/20)*pi, range(1,n_angles+1)))
            plot(angles,Ks)
            xlabel("angle")
            ylabel("K")
            savefig(path_save / "chaos.png")
        return median(Ks)
