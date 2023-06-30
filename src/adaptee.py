import copy
import random
from datetime import datetime

import numpy as np
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid
from sklearn.model_selection._validation import cross_val_score
from sklearn.utils import indexable
from sklearn.utils.validation import _check_fit_params

from hbrkga.brkga_mp_ipr.algorithm import BrkgaMpIpr
from hbrkga.brkga_mp_ipr.enums import Sense
from hbrkga.brkga_mp_ipr.types import BaseChromosome
from hbrkga.brkga_mp_ipr.types_io import load_configuration
from hbrkga.exploitation_method_BO_only_elites import BayesianOptimizerElites


class Decoder:
    """
    A classe Decoder é responsável por fazer a conversão dos parâmetros em indivíduos do algoritmo genético e 
    também por convertê-los de volta.

    Parameters
    ----------
    parameters: <<DIZER QUAIS SÃO OS TIPOS ACEITOS>>
    <<DEFINIR O QUE É ESSE PARÂMETRO (EXEMPLO ABAIXO)>>
    The parameter grid to explore, as a dictionary mapping estimator
    parameters to sequences of allowed values.

    An empty dict signifies default parameters.

    A sequence of dicts signifies a sequence of grids to search, and is
    useful to avoid exploring parameter combinations that make no sense
    or have no effect. See the examples below.
    """

    def __init__(self, parameters: dict, estimator: float, X: float, y: float, cv: dict):
        self._parameters = parameters
        self._estimator = estimator
        self._X = X
        self._y = y
        self._limits = [self._parameters[l] for l in list(self._parameters.keys())]
        self._cv = cv

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        """
        Definição da decodificação, o qual usa como parâmetro o cromossomo base  variável cromossomo e a variável booleana de escrita. 
        O método retorna o resultado da função de avaliação para um dado cromossomo.
        
        Parameters
        
        ----------
        chromosome: BaseChromosome
        <<DFINIR O QUE É ESSE PARÂMETRO>>
        
        rewrite:bool
        <<DEFINIR O QUE É ESSE PARÂMETRO>>
        
        Returns
        
        -------
        score:float
        resultado da função de avaliação para o cromossomo
        """
        return self.score(self.encoder(chromosome))

    def encoder(self, chromosome: BaseChromosome) -> dict:
        """
        A classe encoder é responsável por pegar o valor do cromossomo base repassado na decodificação
        e converter(encode) para o formato dict.
        Para isso são declarados como parâmetros: o tamanho do cromossomo;e  os hiperparâmetros,
        declarando-o com resultado do método deepcopy da cópia,o qual usa os parâmetros de self.
        Após essas declarações, ele percorre o cromossomo usando os genes Idx como marcadores.
        Nisso, os genes encontrados são listados num array de genes Idx, assim como é elaborado
        uma lista array de chaves dos parâmetros de self, e depois os limites, com base nos valores
        do array de chaves já mencionado.
        Para cada valor valor encontrado no array de limites, é listado o valor dos limites até agora.
        Dependendo do tipo(type) de valor encontrado nos limites, o valor do hiperparâmetro criado pode ser: str(string);
        int(integer), contanto que não ultrpasse o valor do limites em 2 unidades; ou mesmo bool(boolean, ou seja, 
        retorna 0 ou 1),ou nenhum deles caso contrário. Após o ciclo de if's terminar, é retornado o valor dos hiperparâmetros.         
        """
        chr_size = len(chromosome)
        hyperparameters = copy.deepcopy(self._parameters)

        for geneIdx in range(chr_size):
            gene = chromosome[geneIdx]
            key = list(self._parameters.keys())[geneIdx]
            limits = self._parameters[key]  # evita for's aninhados
            if type(limits) is np.ndarray:
                limits = limits.tolist()

            if type(limits[0]) is str:
                hyperparameters[key] = limits[round(gene * (len(limits) - 1))]
            elif type(limits[0]) is int and len(limits) > 2:
                hyperparameters[key] = int(limits[round(gene * (len(limits) - 1))])
            elif type(limits[0]) is bool:
                hyperparameters[key] = 1 if limits[0] else 0
            else:
                hyperparameters[key] = (gene * (limits[1] - limits[0])) + limits[0]

        return hyperparameters

    def score(self, hyperparameters: dict) -> float:
        """
        A classe score é responsável por transformar o valor dos hiperparâmetros de self encontrados, no formato dict,
        em formato float para poder indicar a pontuação dos cromossomos encontrados.
        Primeiro, são declarados: o clone do estimador;e os parâmetros configurados do mesmo, com base nos hiperparâmetros.
        Segundo, o try declara para que o clone do estimador usa o método fit,com base nos valores em x e em y do self,
        exceto quando dá erro no valor, o qual nesse caso, o valor retornado é 0.
        Após o try, é retornado o valor de pontuação do cruzamento dos valores encontrados na classe score,
        através do cálculo da média deles.
        """
        estimator_clone = clone(self._estimator)
        estimator_clone.set_params(**hyperparameters)

        try:
            estimator_clone.fit(self._X, self._y)
        except ValueError:
            return 0.0

        # Adicionar o parâmetro scoring para que siga fielmente o SKLearn
        return cross_val_score(estimator_clone, self._X, self._y, cv=self._cv).mean()


class HyperBRKGASearchCV(BaseSearchCV):
    """
    Classe do CV de Busca do HyperBRKGA. Aqui é definido muitas classes  e variáveis importantes para o
    funcionamento do HyperBRKGA.
    A classe de inicialização declara alguns parâmetros e variáveis nos quais o funcionamento do Algoritmo Genético se sustenta,
    bem como o valor inicial de alguns deles.
    Após a declaração destas variáveis, bem como seus valores e sobre quais outras variáveis, métodos e parâmetros eles se sustentam,
    ele inicializa o brkga.
    
    """

    def __init__(
            self,
            estimator,
            *,
            scoring=None,
            n_jobs=None,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch="2*n_jobs",
            error_score=np.nan,
            return_train_score=True,
            parameters,
            data,
            target
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.brkga_config, _ = load_configuration("./hbrkga/config.conf")
        self._parameters = parameters

        self.decoder = Decoder(self._parameters, estimator, data, target, cv)
        elite_number = int(self.brkga_config.elite_percentage * self.brkga_config.population_size)
        self.em_bo = BayesianOptimizerElites(decoder=self.decoder, e=0.3, steps=3, percentage=0.6,
                                             eliteNumber=elite_number)
        chromosome_size = len(self._parameters)
        self.brkga = BrkgaMpIpr(
            decoder=self.decoder,
            sense=Sense.MAXIMIZE,
            seed=random.randint(-10000, 10000),
            chromosome_size=chromosome_size,
            params=self.brkga_config,
            diversity_control_on=True,
            n_close=3,
            exploitation_method=self.em_bo
        )

        self.brkga.initialize()

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """
        A classe fit é responsável por encontrar os valores de aptidão baseado no self, seu valor em x, em y, bem como os grupos 
        e os parâmetros de aptidão para parametrizar os pontuadores.
        Para isso, é declarado, primeiramente, o estimador.
        Depois, ele avalia se o método de pontuação de self é chamável. Em caso positivo, ele é adicionado aos pontuadores.
        Em caso de não existir valor  ou se ele uma instância do método em string, os pontuadores checam o método de pontuação do self, 
        bem como seu estimador.
        Se nenhum dos casos ocorrer, os pontuadores checam o método de pontuação do self e seu estimador na forma multimétrica, 
        e então é ativado o metodo do self de checar suas multimétricas de reaptidão se baseando nos pontuadores como parâmetros.
        Após a verificação, é indexado os valores em x, em y e seus grupos. E depois são checados os parâmetros de aptidão.
        Também são verificados os pontos de origem do cv, bem como as n divisões do cv de origem.

        """
        estimator = self.estimator

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        def evaluate_candidates(candidate_params: dict, cv=None, more_results=None):
        """
        A classe de avaliação dos candidatos. Nele,os cromossomos são avaliados para encontrar potencias candidatos,
        levando-se em consideração os parâmetros de candidatos, a inexistência de valores no cv ou inexistência de mais resultados.
        Primeiramente, a classe começa a marcar o tempo da avaliação, usando data real como referência, depois ele
        faz uma lista dos parâmetros dos candidatos e,depois, os converte em um array contendo todos os parâmetros encontrados.
        E então começa a marcar as gerações da primeira até a décima-primeira. A cada geração imprimida, ele evolui o brkga.
        Ainda no ciclo de cada geração, para cada idx na população no alcance população atual do brkga, ele calcula e marca a pontuação
        de diversidade da mesma. 
        Para valores de verbose do self acima de 2, é imprimida a população, sua pontuação de diversidade e os cromossomos encontrados.
        Para o caso do cromossomo idx estiver no alcance dos cromossomos das populações atuais do brkga, 
        este cromossomo será imprimido, assim como sua aptidão. Depois, são listadas todas as aptidões de cromossomos encontradas.
        Após esses dados, são definidas o melhor cromossomo, bem como a melhor aptidão.
        Após o fim do ciclo de cada geração, são imprimidas: o valor do melhor cromossomo encontrado até agora;
        melhor pontuação(aptidão) até agora; tempo decorrido desde o início da avaliação.
        Após o último ciclo de geração ser concluído, ele imprime os resultados finais, mostrando o melhor cromossomo encontrado , 
        a melhor aptidão, bem como o tempo total decorrido desde que o programa iniciou.
        Após isso, é extendido os parâmetros de candidatos ,os dados e resultados de self são atualizados, 
        e só então o valor de self é retornado.
        """
            start = datetime.now()
            candidate_params = list(candidate_params)
            all_candidate_params = []

            for i in range(1, 11):
                print("\n###############################################")
                print(f"Generation {i}")
                print("")
                self.brkga.evolve()

                for pop_idx in range(len(self.brkga._current_populations)):
                    pop_diversity_score = self.brkga.calculate_population_diversity(pop_idx)
                    if self.verbose > 2:
                        print(f"Population {pop_idx}:")
                        print(f"Population diversity score = {pop_diversity_score}")
                        print("")
                        print("Chromosomes = ")
                        for chromo_idx in range(len(self.brkga._current_populations[pop_idx].chromosomes)):
                            print(f"{chromo_idx} -> {self.brkga._current_populations[pop_idx].chromosomes[chromo_idx]}")
                        print("")
                        print("Fitness = ")
                        for fitness in self.brkga._current_populations[pop_idx].fitness:
                            print(fitness)
                        print("------------------------------")

                best_cost = self.brkga.get_best_fitness()
                best_chr = self.brkga.get_best_chromosome()
                if self.verbose > 2:
                    print(f"{datetime.now()} - Best score so far: {best_cost}")
                    print(f"{datetime.now()} - Best chromosome so far: {best_chr}")
                    print(f"{datetime.now()} - Total time so far: {datetime.now() - start}", flush=True)

            best_cost = self.brkga.get_best_fitness()
            best_chr = self.brkga.get_best_chromosome()
            if self.verbose > 2:
                print("\n###############################################")
                print("Final results:")
                print(f"{datetime.now()} - Best score: {best_cost}")
                print(f"{datetime.now()} - Best chromosome: {best_chr}")
                print(f"Total time = {datetime.now() - start}")

            all_candidate_params.extend(candidate_params)
            self.results = {
                "best_chromosome": best_chr,
                "best_param_decoded": self.decoder.encoder(best_chr),
                "best_param_score": best_cost,
                "total_time": (datetime.now() - start).total_seconds(),
            }

        self._run_search(evaluate_candidates)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = self.results
        self.n_splits_ = n_splits

        return self

    def _run_search(self, evaluate_candidates: dict):
        """
        A última classe do HyperBrkgaCV,cujo proósito é rodar a busca. Para isso, so era preciso avaliar o candidatos, 
        usando como parâmetro a grade de parâmetros do self
        """
        evaluate_candidates(ParameterGrid(self._parameters))
