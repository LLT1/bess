from bess.cbess import pywrap_bess
import numpy as np
import math

class bess_base:
    def __init__(self, algrithm_type, model_type, path_type, max_iter=20, is_warm_start=False,
                 coef0=0, sequence=None, s_min=None, s_max=None, K_max=None, epsilon=0.0001, ic_type="aic",
                 is_cv=False, K=5):
        self.algrithm_type = algrithm_type
        self.model_type = model_type
        self.path_type = path_type
        self.max_iter = max_iter
        self.is_warm_start = is_warm_start
        self.coef0 = coef0
        self.sequence = sequence
        self.s_min = s_min
        self.s_max = s_max
        self.K_max = K_max
        self.epsilon = epsilon
        self.ic_type = ic_type
        self.is_cv = is_cv
        self.K = K
        self.path_len = None
        self.p = None

        self.beta_out = None
        self.coef0_out = None
        self.train_loss_out = None
        self.ic_out = None
        # self.nullloss_out = None
        # self.aic_out = None
        # self.bic_out = None
        # self.gic_out = None
        # self.A_out = None
        # self.l_out = None

        self.arg_check()

    def arg_check(self):
        # print("arg_check")
        if self.algrithm_type == "Pdas":
            self.algrithm_type = int(1)
        elif self.algrithm_type == "GroupPdas":
            self.algrithm_type = int(2)
        else:
            raise ValueError("algrithm_type should not be " + str(self.algrithm_type))

        if self.model_type == "Lm":
            self.model_type = int(1)
        elif self.model_type == "Logistic":
            # print("GLM")
            self.model_type = int(2)
        else:
            raise ValueError("model_type should not be " + str(self.model_type))

        if self.path_type == "seq":
            # if self.sequence is None:
            #     raise ValueError(
            #         "When you choose path_type = sequence-search, the parameter \'sequence\' should be given.")
            self.path_type = 1


        elif self.path_type == "gs":
            # if self.s_min is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'s_min\' should be given.")
            #
            # if self.s_max is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'s_max\' should be given.")
            #
            # if self.K_max is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'K_max\' should be given.")
            #
            # if self.epsilon is None:
            #     raise ValueError(
            #         " When you choose path_type = golden-section-search, the parameter \'epsilon\' should be given.")

            self.path_type = 2


        if self.ic_type == "aic":
            self.ic_type = 1
        elif self.ic_type == "bic":
            self.ic_type = 2
        elif self.ic_type == "gic":
            self.ic_type = 3
        else:
            raise ValueError("ic_type should be \"aic\", \"bic\" or \"gic\"")
         
    def fit(self, X, y, is_weight=False, is_normal=False, weight=None, state=None):
        self.p = X.shape[1]
        n = X.shape[0]
        p = X.shape[1]
        if n != y.size:
            raise ValueError("X.shape(0) should be equal to y.size")

        if is_weight:
            if weight is None:
                raise ValueError("When you choose is_weight is True, the parameter weight should be given")
            else:
                if n != weight.size:
                    raise ValueError("X.shape(0) should be equal to weight.size")
        else:
            weight = np.ones(n) / n

        # To do
        if state is None:
            state = np.ones(n) / n

        # path parameter
        if self.path_type == 1:
            if self.sequence is None:
                self.sequence = [(i+1) for i in range(p)]
            self.s_min = 0
            self.s_max = 0
            self.K_max = 0
            self.path_len = int(len(self.sequence))
        else:
            self.sequence = [1]
            if self.s_min is None:
                self.s_min = 1
            if self.s_max is None:
                self.s_max = p
            if self.K_max is None:
                self.K_max = int(math.log(p, 2/(math.sqrt(5) - 1)))
            self.path_len = self.K_max + 2

        # print("run")
        # print("model type"+ str(self.model_type))
        result = pywrap_bess(X, y, self.data_type, weight,
                             is_normal,
                             self.algrithm_type, self.model_type, self.max_iter,
                             self.path_type, self.is_warm_start,
                             self.ic_type, self.is_cv, self.K,
                             state,
                             self.coef0,
                             self.sequence,
                             self.s_min, self.s_max, self.K_max, self.epsilon,
                             p,
                             1, 1, 1, 1, 1, 1, p
                             )

        self.beta_out = result[0]
        self.coef0_out = result[1]
        self.train_loss_out = result[2]
        self.ic_out = result[3]
        # self.nullloss_out = result[3]
        # self.aic_out = result[4]
        # self.bic_out = result[5]
        # self.gic_out = result[6]
        # self.A_out = result[7]
        # self.l_out = result[8]

    def predict(self, X):
        if X.shape[1] != self.p:
            raise ValueError("X.shape[1] should be " + str(self.p))

        # beta = []
        # coef = self.coef0_out[self.l_out - 1]
        # for i in range(self.p):
        #     beta.append(self.beta_out[self.l_out - 1 + i * self.path_len])

        if self.model_type == 1:
            coef0 = np.ones(X.shape[0]) * self.coef0_out
            return np.dot(X, self.beta_out) + coef0
        elif self.model_type == 2:
            coef0 = np.ones(X.shape[0]) * self.coef0_out
            xbeta = np.dot(X, self.beta_out) + coef0

            y = np.zeros(xbeta.size)
            y[xbeta > 0] = 1

            xbeta[xbeta > 25] = 25
            xbeta[xbeta < -25] = -25
            xbeta_exp = np.exp(xbeta)
            pr = xbeta_exp / (xbeta_exp + 1)

            result = dict()
            result["Y"] = y
            result["pr"] = pr
            return result


class PdasLm(bess_base):
    """
        PdasLm

        Parameters
        ----------
        max_iter=20, is_warm_start=False, sequence, ic_type=1, is_cv=False, K=5

        max_iter : int, optional
            Max iteration time in PDAS.
            Default: max_iter = 20.
        is_warm_start : bool, optional
            When search the best sparsity,whether use the last parameter as the initial parameter for the next search.
            Default:is_warm_start = False.
        path_type : {"seq", "gs"}
            The method we use to search the sparsity。
        sequence : array_like, optional
            The  sparsity list for searching. If choose path_type = "seq", we prefer you to give the sequence.If not
            given, we will search all the sparsity([1,2,...,p],p=X.shape[1]).
            Default: sequence = None.
        s_min : int, optional
            The lower bound of golden-section-search for sparsity searching.If not given, we will set s_min = 1.
            Default: s_min = None.
        s_max : int, optional
            The higher bound of golden-section-search for sparsity searching.If not given, we will set s_max = p(p = X.shape[1]).
            Default: s_max = None.
        K_max : int, optional
            The search times of golden-section-search for sparsity searching.If not given, we will set K_max = int(log(p, 2/(math.sqrt(5) - 1))).
            Default: K_max = None.
        epsilon : double, optional
            The stop condition of golden-section-search for sparsity searching.
            Default: epsilon = 0.0001.
        ic_type : {'aic', 'bic', 'gic'}, optional
            The metric when choose the best sparsity.
            Input must be one of the set above. Default: ic_type = "aic".
        is_cv : bool, optional
            Use the Cross-validation method to caculate the loss.
            Default: is_cv = False.
        K : int optional
            The folds number when Use the Cross-validation method to caculate the loss.
            Default: K = 5.

        References
        ----------
        - Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear,
         logistic and coxph models]

        Examples
        --------
        >>> from bess.bess import *
        >>> import numpy as np
        >>> np.random.seed(123)   # fix seed to get the same result
        >>> # Data information
        >>> train_X = np.random.normal(0, 1, 10 * 5).reshape((10, 5))     # train_x
        >>> train_y = np.random.normal(0, 1, 10)                          # train_y
        >>> test_X = np.random.normal(0, 1, 10 * 5).reshape((10, 5))      # test_x
        >>> # Model information.
        >>> model = PdasLm()
        >>> # Fit model.
        >>> model.fit(train_X, train_y)
        >>> # Predict.
        >>> model.predict(test_X)
        [0.8487986542970937, -0.3262548528320405, -1.2427679709665753, -1.1018898695690327, -1.1589700150336006, 0.9170341511542658, -0.7894714094101583, -0.40658125651728766, -0.5161022067202307, -0.17317826396384567]

        """
    def __init__(self, max_iter=20, path_type="seq", is_warm_start=False, sequence=None, s_min=None, s_max=None,
                 K_max=None, epsilon=0.0001, ic_type="aic", is_cv=False, K=5
                 ):
        super(PdasLm, self).__init__(
            algrithm_type="Pdas", model_type="Lm", path_type=path_type, max_iter=max_iter,
            is_warm_start=is_warm_start, sequence=sequence, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, ic_type=ic_type, is_cv=is_cv, K=K)
        self.data_type = 1


class PdasLogistic(bess_base):
    """
        PdasLogistic

        Parameters
        ----------
        max_iter=20, is_warm_start=False, sequence, ic_type=1, is_cv=False, K=5

        max_iter : int, optional
            Max iteration time in PDAS.
            Default: max_iter = 20.
        is_warm_start : bool, optional
            When search the best sparsity,whether use the last parameter as the initial parameter for the next search.
            Default:is_warm_start = False.
        path_type : {"seq", "gs"}
            The method we use to search the sparsity。
        sequence : array_like, optional
            The  sparsity list for searching. If choose path_type = "seq", we prefer you to give the sequence.If not
            given, we will search all the sparsity([1,2,...,p],p=X.shape[1]).
            Default: sequence = None.
        s_min : int, optional
            The lower bound of golden-section-search for sparsity searching.If not given, we will set s_min = 1.
            Default: s_min = None.
        s_max : int, optional
            The higher bound of golden-section-search for sparsity searching.If not given, we will set s_max = p(p = X.shape[1]).
            Default: s_max = None.
        K_max : int, optional
            The search times of golden-section-search for sparsity searching.If not given, we will set K_max = int(log(p, 2/(math.sqrt(5) - 1))).
            Default: K_max = None.
        epsilon : double, optional
            The stop condition of golden-section-search for sparsity searching.
            Default: epsilon = 0.0001.
        ic_type : {'aic', 'bic', 'gic'}, optional
            The metric when choose the best sparsity.
            Input must be one of the set above. Default: ic_type = "aic".
        is_cv : bool, optional
            Use the Cross-validation method to caculate the loss.
            Default: is_cv = False.
        K : int optional
            The folds number when Use the Cross-validation method to caculate the loss.
            Default: K = 5.

        References
        ----------
        - Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear,
         logistic and coxph models]

        Examples
        --------
        >>> from bess.bess import *
        >>> import numpy as np
        >>> np.random.seed(123)   # fix seed to get the same result
        >>> # Data information
        >>> train_X = np.random.normal(0, 1, 10 * 5).reshape((10, 5))     # train_x
        >>> train_y = np.random.randint(0, 2, 10)                         # train_y
        >>> test_X = np.random.normal(0, 1, 10 * 5).reshape((10, 5))      # test_x
        >>> # Model information.
        >>> model = PdasLogistic(path_type="gs")
        >>> # Fit model.
        >>> model.fit(train_X, train_y)
        >>> # Predict.
        >>> model.predict(test_X)
        {'Y': array([1., 1., 1., 1., 1., 0., 0., 1., 1., 1.]),
        'pr': array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 1.38879439e-11, 1.38879439e-11, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00])}

    """

    def __init__(self, max_iter=20, path_type="seq", is_warm_start=False, coef0=0, sequence=None, s_min=None, s_max=None,
                 K_max=None, epsilon=0.0001, ic_type="aic", is_cv=False, K=5
                 ):
        super(PdasLogistic, self).__init__(
            algrithm_type="Pdas", model_type="Logistic", path_type=path_type, max_iter=max_iter,
            is_warm_start=is_warm_start, coef0=coef0, sequence=sequence, s_min=s_min, s_max=s_max, K_max=K_max,
            epsilon=epsilon, ic_type=ic_type, is_cv=is_cv, K=K)
        self.data_type = 2


# class SGROUP_LM(bess_base):
#
#     def __init__(self, max_iter=20, is_warm_start=False, sequence=None, ic_type=1, is_cv=False, K=5
#                  ):
#         super(SGROUP_LM, self).__init__(
#             algrithm_type="PDAS", model_type="LM", path_type="sequence", max_iter=max_iter,
#             is_warm_start=is_warm_start, sequence=sequence, ic_type=ic_type, is_cv=is_cv, K=K)
#
#
# class GGROUP_LM(bess_base):
#     def __init__(self, max_iter=20, is_warm_start=False, s_min=1, s_max=10, K_max=10, epsilon=0.0001,
#                  ic_type=1, is_cv=False, K=5):
#         super(GGROUP_LM, self).__init__(
#             algrithm_type="PDAS", model_type="LM", path_type="gs", max_iter=max_iter,
#             is_warm_start=is_warm_start, s_min=s_min, s_max=s_max, K_max=K_max, epsilon=epsilon,
#             ic_type=ic_type, is_cv=is_cv, K=K)

# class SGROUP_GLM(bess_base):
#
#     def __init__(self, max_iter=20, is_warm_start=False,coef0=0, sequence=None, ic_type=1, is_cv=False, K=5
#                  ):
#         super(SGROUP_GLM, self).__init__(
#             algrithm_type="PDAS", model_type="GLM", path_type="sequence", max_iter=max_iter,
#             is_warm_start=is_warm_start, coef0=coef0, sequence=sequence, ic_type=ic_type, is_cv=is_cv, K=K)
#
#
# class GGROUP_GLM(bess_base):
#     def __init__(self, max_iter=20, is_warm_start=False, coef0=0, s_min=1, s_max=10, K_max=10, epsilon=0.0001,
#                  ic_type=1, is_cv=False, K=5):
#         super(GGROUP_GLM, self).__init__(
#             algrithm_type="PDAS", model_type="GLM", path_type="gs", max_iter=max_iter,
#             is_warm_start=is_warm_start, coef0=coef0, s_min=s_min, s_max=s_max, K_max=K_max, epsilon=epsilon,
#             ic_type=ic_type, is_cv=is_cv, K=K)



