#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:47:08 2023

@author: rcharayr
"""

import unittest

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

import numpy.linalg as npl

from smt.applications import NestedLHS
from smt.applications.mfk import MFK
from smt.applications.mfkpls import MFKPLS
from smt.applications.mixed_integer import (
    MixedIntegerSamplingMethod,
)
from smt.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
)
from smt.sampling_methods import LHS
from smt.surrogate_models import (
    KPLS,
    KRG,
    MixIntKernelType,
)


class TestMFKmixed(unittest.TestCase):
    # useful functions (g and h)
    def g_ZDT(self, x):
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        if x[5] == 0:  # 0 -> blue
            x5 = 0.1
        if x[5] == 1:  # 1 -> red
            x5 = 0.5
        if x[5] == 2:  # 2 -> green
            x5 = 0.75

        out = 1 + (9 / (6 - 1)) * (x1 + x2 + x3 + x4 + x5)
        return out

    def g_DTLZ5(self, x):
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        if x[5] == 0:  # 0 -> blue
            x5 = 0.1
        if x[5] == 1:  # 1 -> red
            x5 = 0.5
        if x[5] == 2:  # 2 -> green
            x5 = 0.75

        out = (
            (x1 - 0.5) ** 2
            + (x2 - 0.5) ** 2
            + (x3 - 0.5) ** 2
            + (x4 - 0.5) ** 2
            + (x5 - 0.5) ** 2
        )
        return out

    def h_ZDT1(self, x):
        x0 = x[0]
        out = 1 - np.sqrt(x0 / self.g_ZDT(x))
        return out

    def h_ZDT2(self, x):
        x0 = x[0]
        out = 1 - (x0 / self.g_ZDT(x)) ** 2
        return out

    def h_ZDT3(self, x):
        x0 = x[0]
        out = (
            1
            - np.sqrt(x0 / self.g_ZDT(x))
            - (x0 / self.g_ZDT(x)) * np.sin(10 * np.pi * x0)
        )
        return out

    # grouped functions
    def ZDT1_HF_mixt(self, x):
        x0 = x[0]
        y1 = x0
        y2 = self.g_ZDT(x) * self.h_ZDT1(x)

        fail = False
        return [y1, y2], fail

    def ZDT1_LF_mixt(self, x):
        x0 = x[0]
        y1 = 0.9 * x0 + 0.1
        y2 = (0.8 * self.g_ZDT(x) - 0.2) * (1.2 * self.h_ZDT1(x) + 0.2)

        fail = False
        return [y1, y2], fail

    ###############################################################################

    def ZDT2_HF_mixt(self, x):
        x0 = x[0]
        y1 = x0
        y2 = self.g_ZDT(x) * self.h_ZDT2(x)

        fail = False
        return [y1, y2], fail

    def ZDT2_LF_mixt(self, x):
        x0 = x[0]
        y1 = 0.8 * x0 + 0.2
        y2 = (0.9 * self.g_ZDT(x) + 0.2) * (1.1 * self.h_ZDT2(x) - 0.2)

        fail = False
        return [y1, y2], fail

    ###############################################################################

    def ZDT3_HF_mixt(self, x):
        x0 = x[0]
        y1 = x0
        y2 = self.g_ZDT(x) * self.h_ZDT3(x)
        fail = False
        return [y1, y2], fail

    def ZDT3_LF_mixt(self, x):
        x0 = x[0]
        y1 = 0.75 * x0 + 0.25
        y2 = self.g_ZDT(x) * (1.25 * self.h_ZDT3(x) - 0.25)
        fail = False
        return [y1, y2], fail

    ###############################################################################

    def DTLZ5_HF_mixt(self, x):
        x0 = x[0]
        y1 = (1 + self.g_DTLZ5(x)) * np.cos(0.5 * np.pi * x0)
        y2 = (1 + self.g_DTLZ5(x)) * np.sin(0.5 * np.pi * x0)
        fail = False
        return [y1, y2], fail

    def DTLZ5_LF_mixt(self, x):
        x0 = x[0]
        y1 = (1 + 0.8 * self.g_DTLZ5(x)) * np.cos(0.5 * np.pi * x0)
        y2 = (1 + 1.1 * self.g_DTLZ5(x)) * np.sin(0.5 * np.pi * x0)

        fail = False
        return [y1, y2], fail

    ###############################################################################

    def run_mfk_mixed_example(self):
        import matplotlib.pyplot as plt

        # KRG_METHODS = ["krg", "kpls", "mfk", "mfkpls"]
        # KRG_METHODS = ["krg"]
        # KRG_METHODS = ["kpls"]
        KRG_METHODS = ["mfk"]
        # KRG_METHODS = ["mfkpls"]

        pbm = "ZDT1"

        if pbm == "ZDT1":
            HF_fun = self.ZDT1_HF_mixt
            LF_fun = self.ZDT1_LF_mixt
        elif pbm == "ZDT2":
            HF_fun = self.ZDT2_HF_mixt
            LF_fun = self.ZDT2_LF_mixt
        elif pbm == "ZDT3":
            HF_fun = self.ZDT3_HF_mixt
            LF_fun = self.ZDT3_LF_mixt
        elif pbm == "DTLZ5":
            HF_fun = self.DTLZ5_HF_mixt
            LF_fun = self.DTLZ5_LF_mixt

        # ------------------------------------------------------------------------------

        # Instantiate the design space with all its design variables:
        ds = DesignSpace(
            [
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                # OrdinalVariable(['0', '1']),  # x4 ordinal: 0 and 1; order is relevant
                IntegerVariable(0, 1),  # x4 integer between 0 and 1
                CategoricalVariable(
                    ["blue", "red", "green"]
                ),  # x5 categorical: blue, red or green; order is not relevant
            ]
        )

        # ------------------------------------------------------------------------------

        # Validation data:
        n_valid = 100  # validation set size
        ds.seed = 42
        samp = MixedIntegerSamplingMethod(LHS, ds, criterion="ese", seed=ds.seed)
        x_valid, is_acting_valid = samp(n_valid, return_is_acting=True)

        y1_valid = np.zeros(n_valid)  # obj 1
        y2_valid = np.zeros(n_valid)  # obj 2
        y1_valid_LF = np.zeros(n_valid)  # obj 1
        y2_valid_LF = np.zeros(n_valid)  # obj 2
        for i in range(n_valid):
            res = HF_fun(x_valid[i, :])
            y1_valid[i] = res[0][0]  # obj 1
            y2_valid[i] = res[0][1]  # obj 2
            res_LF = LF_fun(x_valid[i, :])
            y1_valid_LF[i] = res_LF[0][0]  # obj 1
            y2_valid_LF[i] = res_LF[0][1]  # obj 2
        # ------------------------------------------------------------------------------

        # HF training data:
        n_train_HF = 20  # training set size
        n_train_LF = 40
        nlhs = NestedLHS(nlevel=2, design_space=ds)
        xt_LF, xt_HF = nlhs(n_train_HF)

        y1t_HF = np.zeros(n_train_HF)  # obj 1
        y2t_HF = np.zeros(n_train_HF)  # obj 2
        for i in range(n_train_HF):
            res = HF_fun(xt_HF[i, :])
            y1t_HF[i] = res[0][0]  # obj 1
            y2t_HF[i] = res[0][1]  # obj 2

        # ------------------------------------------------------------------------------

        for krg_method in KRG_METHODS:
            if "mfk" in krg_method:  # if multifi compute LF validation data
                y1t_LF = np.zeros(n_train_LF)  # obj 1
                y2t_LF = np.zeros(n_train_LF)  # obj 2
                for i in range(n_train_LF):
                    res_LF = LF_fun(xt_LF[i, :])
                    y1t_LF[i] = res_LF[0][0]  # obj 1
                    y2t_LF[i] = res_LF[0][1]  # obj 2

            if krg_method == "krg":
                print("KRG")
                ###########################
                # Mono-fidelity KRG
                ###########################

                sm1 = KRG(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )
                sm2 = KRG(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_HF, y1t_HF)
                # obj 2
                sm2.set_training_values(xt_HF, y2t_HF)

            if krg_method == "kpls":
                print("KPLS")
                ###########################
                # Mono-fidelity KPLS
                ###########################

                sm1 = KPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )
                sm2 = KPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_HF, y1t_HF)
                # obj 2
                sm2.set_training_values(xt_HF, y2t_HF)

            if krg_method == "mfk":
                print("MFK")
                ###########################
                # MFK
                ###########################

                sm1 = MFK(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                    hyper_opt="Cobyla",
                )
                sm2 = MFK(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                    hyper_opt="Cobyla",
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_LF[:, :], y1t_LF, name=0)
                sm1.set_training_values(xt_HF[:, :], y1t_HF)
                # obj 2
                sm2.set_training_values(xt_LF[:, :], y2t_LF, name=0)
                sm2.set_training_values(xt_HF[:, :], y2t_HF)

            if krg_method == "mfkpls":
                print("MFKPLS")
                ###########################
                # MFKKPLS
                ##########################

                sm1 = MFKPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                    hyper_opt="Cobyla",
                )
                sm2 = MFKPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                    hyper_opt="Cobyla",
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_LF[:, :], y1t_LF, name=0)
                sm1.set_training_values(xt_HF[:, :], y1t_HF)
                # obj 2
                sm2.set_training_values(xt_LF[:, :], y2t_LF, name=0)
                sm2.set_training_values(xt_HF[:, :], y2t_HF)

            # Train the models:
            import time

            t0 = time.time()
            sm1.train()
            t1 = time.time()
            sm2.train()
            t2 = time.time()
            print("t1-t0=", t1 - t0)
            print("t2-t1=", t2 - t1)

            # Compute errors:
            # obj 1
            y1_predict = np.ravel(sm1.predict_values(x_valid))
            if "mfk" in krg_method:
                y1_predict_LF = np.ravel(sm1._predict_intermediate_values(x_valid, 1))
                _y1_var_predict_LF = np.ravel(
                    sm1.predict_variances_all_levels(x_valid)[0]
                )
            y1_var_predict = np.ravel(sm1.predict_variances(x_valid))
            _y1_sd_predict = np.sqrt(y1_var_predict)
            # obj 2
            y2_predict = np.ravel(sm2.predict_values(x_valid))
            if "mfk" in krg_method:
                y2_predict_LF = np.ravel(sm2._predict_intermediate_values(x_valid, 1))
                _y2_var_predict_LF = np.ravel(
                    sm2.predict_variances_all_levels(x_valid)[0]
                )
            y2_var_predict = np.ravel(sm2.predict_variances(x_valid))
            _y2_sd_predict = np.sqrt(y2_var_predict)

            # obj 1:
            # print("y1_predict - y1_valid =", y1_predict - y1_valid)
            print("norme2(y1_predict - y1_valid) =", npl.norm(y1_predict - y1_valid, 2))
            if "mfk" in krg_method:
                print(
                    "norme2(y1_predict_LF - y1_valid_LF) =",
                    npl.norm(y1_predict_LF - y1_valid_LF, 2),
                )
            # print("y1_sd_predict =", y1_sd_predict)

            # obj 2:
            # print("y2_predict - y2_valid =", y2_predict - y2_valid)
            print("norme2(y2_predict - y2_valid) =", npl.norm(y2_predict - y2_valid, 2))
            if "mfk" in krg_method:
                print(
                    "norme2(y2_predict_LF - y2_valid_LF) =",
                    npl.norm(y2_predict_LF - y2_valid_LF, 2),
                )
            # print("y2_sd_predict =", y2_sd_predict)

            # PLOTS:
            plt.figure()
            plt.title("y1")
            plt.xlabel("y1_valid")
            plt.ylabel("y1_predict")
            plt.scatter(y1_valid, y1_predict)
            plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red")
            plt.show()

            plt.figure()
            plt.title("y2")
            plt.xlabel("y2_valid")
            plt.ylabel("y2_predict")
            plt.scatter(y2_valid, y2_predict)
            plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
            plt.show()

            if "mfk" in krg_method:
                plt.figure()
                plt.title("y1")
                plt.xlabel("y1_valid_LF")
                plt.ylabel("y1_predict_LF")
                plt.scatter(y1_valid_LF, y1_predict_LF)
                plt.plot(
                    np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red"
                )
                plt.show()

                plt.figure()
                plt.title("y2")
                plt.xlabel("y2_valid_LF")
                plt.ylabel("y2_predict_LF")
                plt.scatter(y2_valid_LF, y2_predict_LF)
                plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
                plt.show()

            # ------------------------------------------------------------------------------

            # PLOTS tests:
            plt.figure()
            plt.title("y1")
            plt.xlabel("y1t_HF")
            plt.ylabel("np.ravel(sm1.predict_values(xt_HF))")
            plt.scatter(y1t_HF, np.ravel(sm1.predict_values(xt_HF)))
            plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red")
            plt.show()

            plt.figure()
            plt.title("y2")
            plt.xlabel("y2t_HF")
            plt.ylabel("np.ravel(sm2.predict_values(xt_HF))")
            plt.scatter(y2t_HF, np.ravel(sm2.predict_values(xt_HF)))
            plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
            plt.show()

            if "mfk" in krg_method:
                plt.figure()
                plt.title("y1")
                plt.xlabel("y1t_LF")
                plt.ylabel("np.ravel(sm1._predict_intermediate_values(xt_LF,1))")
                plt.scatter(
                    y1t_LF, np.ravel(sm1._predict_intermediate_values(xt_LF, 1))
                )
                plt.plot(
                    np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red"
                )
                plt.show()

                plt.figure()
                plt.title("y2")
                plt.xlabel("y2t_LF")
                plt.ylabel("np.ravel(sm2._predict_intermediate_values(xt_LF,1))")
                plt.scatter(
                    y2t_LF, np.ravel(sm2._predict_intermediate_values(xt_LF, 1))
                )
                plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
                plt.show()
                # ------------------------------------------------------------------------------

    def run_mfkpls_mixed_example(self):
        import matplotlib.pyplot as plt

        # KRG_METHODS = ["krg", "kpls", "mfk", "mfkpls"]
        # KRG_METHODS = ["krg"]
        # KRG_METHODS = ["kpls"]
        # KRG_METHODS = ["mfk"]
        KRG_METHODS = ["mfkpls"]

        pbm = "ZDT1"

        if pbm == "ZDT1":
            HF_fun = self.ZDT1_HF_mixt
            LF_fun = self.ZDT1_LF_mixt
        elif pbm == "ZDT2":
            HF_fun = self.ZDT2_HF_mixt
            LF_fun = self.ZDT2_LF_mixt
        elif pbm == "ZDT3":
            HF_fun = self.ZDT3_HF_mixt
            LF_fun = self.ZDT3_LF_mixt
        elif pbm == "DTLZ5":
            HF_fun = self.DTLZ5_HF_mixt
            LF_fun = self.DTLZ5_LF_mixt

        # ------------------------------------------------------------------------------

        # Instantiate the design space with all its design variables:
        ds = DesignSpace(
            [
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                FloatVariable(0, 1),  # x0 continuous between 0 and 1
                # OrdinalVariable(['0', '1']),  # x4 ordinal: 0 and 1; order is relevant
                IntegerVariable(0, 1),  # x4 integer between 0 and 1
                CategoricalVariable(
                    ["blue", "red", "green"]
                ),  # x5 categorical: blue, red or green; order is not relevant
            ]
        )

        # ------------------------------------------------------------------------------

        # Validation data:
        n_valid = 100  # validation set size
        x_valid, is_acting_valid = ds.sample_valid_x(n_valid)

        y1_valid = np.zeros(n_valid)  # obj 1
        y2_valid = np.zeros(n_valid)  # obj 2
        y1_valid_LF = np.zeros(n_valid)  # obj 1
        y2_valid_LF = np.zeros(n_valid)  # obj 2
        for i in range(n_valid):
            res = HF_fun(x_valid[i, :])
            y1_valid[i] = res[0][0]  # obj 1
            y2_valid[i] = res[0][1]  # obj 2
            res_LF = LF_fun(x_valid[i, :])
            y1_valid_LF[i] = res_LF[0][0]  # obj 1
            y2_valid_LF[i] = res_LF[0][1]  # obj 2
        # ------------------------------------------------------------------------------

        # HF training data:
        n_train_HF = 20  # training set size
        xt_HF, is_acting_t_HF = ds.sample_valid_x(n_train_HF)

        y1t_HF = np.zeros(n_train_HF)  # obj 1
        y2t_HF = np.zeros(n_train_HF)  # obj 2
        for i in range(n_train_HF):
            res = HF_fun(xt_HF[i, :])
            y1t_HF[i] = res[0][0]  # obj 1
            y2t_HF[i] = res[0][1]  # obj 2

        # ------------------------------------------------------------------------------

        for krg_method in KRG_METHODS:
            if "mfk" in krg_method:  # if multifi compute LF validation data
                n_train_LF = 40  # training set size   IF MULTIFI
                sampling = MixedIntegerSamplingMethod(LHS, ds, criterion="ese")
                xt_LF = np.concatenate(
                    (xt_HF, sampling.expand_lhs(xt_HF, n_train_LF - n_train_HF)), axis=0
                )

                y1t_LF = np.zeros(n_train_LF)  # obj 1
                y2t_LF = np.zeros(n_train_LF)  # obj 2
                for i in range(n_train_LF):
                    res_LF = LF_fun(xt_LF[i, :])
                    y1t_LF[i] = res_LF[0][0]  # obj 1
                    y2t_LF[i] = res_LF[0][1]  # obj 2

            if krg_method == "krg":
                print("KRG")
                ###########################
                # Mono-fidelity KRG
                ###########################

                sm1 = KRG(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )
                sm2 = KRG(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_HF, y1t_HF)
                # obj 2
                sm2.set_training_values(xt_HF, y2t_HF)

            if krg_method == "kpls":
                print("KPLS")
                ###########################
                # Mono-fidelity KPLS
                ###########################

                sm1 = KPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )
                sm2 = KPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_HF, y1t_HF)
                # obj 2
                sm2.set_training_values(xt_HF, y2t_HF)

            if krg_method == "mfk":
                print("MFK")
                ###########################
                # MFK
                ###########################

                sm1 = MFK(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )
                sm2 = MFK(
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_LF[:, :], y1t_LF, name=0)
                sm1.set_training_values(xt_HF[:, :], y1t_HF)
                # obj 2
                sm2.set_training_values(xt_LF[:, :], y2t_LF, name=0)
                sm2.set_training_values(xt_HF[:, :], y2t_HF)

            if krg_method == "mfkpls":
                print("MFKPLS")
                ###########################
                # MFKKPLS
                ##########################

                sm1 = MFKPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )
                sm2 = MFKPLS(
                    n_comp=3,
                    design_space=ds,
                    theta0=[1e-2],
                    print_prediction=False,
                    corr="squar_exp",
                    categorical_kernel=MixIntKernelType.CONT_RELAX,
                )

                # Set training data
                # obj 1
                sm1.set_training_values(xt_LF[:, :], y1t_LF, name=0)
                sm1.set_training_values(xt_HF[:, :], y1t_HF)
                # obj 2
                sm2.set_training_values(xt_LF[:, :], y2t_LF, name=0)
                sm2.set_training_values(xt_HF[:, :], y2t_HF)

            # Train the models:
            import time

            t0 = time.time()
            sm1.train()
            t1 = time.time()
            sm2.train()
            t2 = time.time()
            print("t1-t0=", t1 - t0)
            print("t2-t1=", t2 - t1)

            # Compute errors:
            # obj 1
            y1_predict = np.ravel(sm1.predict_values(x_valid))
            if "mfk" in krg_method:
                y1_predict_LF = np.ravel(sm1._predict_intermediate_values(x_valid, 1))
                _y1_var_predict_LF = np.ravel(
                    sm1.predict_variances_all_levels(x_valid)[0]
                )
            y1_var_predict = np.ravel(sm1.predict_variances(x_valid))
            _y1_sd_predict = np.sqrt(y1_var_predict)
            # obj 2
            y2_predict = np.ravel(sm2.predict_values(x_valid))
            if "mfk" in krg_method:
                y2_predict_LF = np.ravel(sm2._predict_intermediate_values(x_valid, 1))
                _y2_var_predict_LF = np.ravel(
                    sm2.predict_variances_all_levels(x_valid)[0]
                )
            y2_var_predict = np.ravel(sm2.predict_variances(x_valid))
            _y2_sd_predict = np.sqrt(y2_var_predict)

            # obj 1:
            # print("y1_predict - y1_valid =", y1_predict - y1_valid)
            print("norme2(y1_predict - y1_valid) =", npl.norm(y1_predict - y1_valid, 2))
            if "mfk" in krg_method:
                print(
                    "norme2(y1_predict_LF - y1_valid_LF) =",
                    npl.norm(y1_predict_LF - y1_valid_LF, 2),
                )
            # print("y1_sd_predict =", y1_sd_predict)

            # obj 2:
            # print("y2_predict - y2_valid =", y2_predict - y2_valid)
            print("norme2(y2_predict - y2_valid) =", npl.norm(y2_predict - y2_valid, 2))
            if "mfk" in krg_method:
                print(
                    "norme2(y2_predict_LF - y2_valid_LF) =",
                    npl.norm(y2_predict_LF - y2_valid_LF, 2),
                )
            # print("y2_sd_predict =", y2_sd_predict)

            # PLOTS:
            plt.figure()
            plt.title("y1")
            plt.xlabel("y1_valid")
            plt.ylabel("y1_predict")
            plt.scatter(y1_valid, y1_predict)
            plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red")
            plt.show()

            plt.figure()
            plt.title("y2")
            plt.xlabel("y2_valid")
            plt.ylabel("y2_predict")
            plt.scatter(y2_valid, y2_predict)
            plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
            plt.show()

            if "mfk" in krg_method:
                plt.figure()
                plt.title("y1")
                plt.xlabel("y1_valid_LF")
                plt.ylabel("y1_predict_LF")
                plt.scatter(y1_valid_LF, y1_predict_LF)
                plt.plot(
                    np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red"
                )
                plt.show()

                plt.figure()
                plt.title("y2")
                plt.xlabel("y2_valid_LF")
                plt.ylabel("y2_predict_LF")
                plt.scatter(y2_valid_LF, y2_predict_LF)
                plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
                plt.show()

            # ------------------------------------------------------------------------------

            # PLOTS tests:
            plt.figure()
            plt.title("y1")
            plt.xlabel("y1t_HF")
            plt.ylabel("np.ravel(sm1.predict_values(xt_HF))")
            plt.scatter(y1t_HF, np.ravel(sm1.predict_values(xt_HF)))
            plt.plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red")
            plt.show()

            plt.figure()
            plt.title("y2")
            plt.xlabel("y2t_HF")
            plt.ylabel("np.ravel(sm2.predict_values(xt_HF))")
            plt.scatter(y2t_HF, np.ravel(sm2.predict_values(xt_HF)))
            plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
            plt.show()

            if "mfk" in krg_method:
                plt.figure()
                plt.title("y1")
                plt.xlabel("y1t_LF")
                plt.ylabel("np.ravel(sm1._predict_intermediate_values(xt_LF,1))")
                plt.scatter(
                    y1t_LF, np.ravel(sm1._predict_intermediate_values(xt_LF, 1))
                )
                plt.plot(
                    np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100), color="red"
                )
                plt.show()

                plt.figure()
                plt.title("y2")
                plt.xlabel("y2t_LF")
                plt.ylabel("np.ravel(sm2._predict_intermediate_values(xt_LF,1))")
                plt.scatter(
                    y2t_LF, np.ravel(sm2._predict_intermediate_values(xt_LF, 1))
                )
                plt.plot(np.linspace(0, 9, 100), np.linspace(0, 9, 100), color="red")
                plt.show()
                # ------------------------------------------------------------------------------

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_mfkpls_mixed(self):
        self.run_mfk_mixed_example()
        self.run_mfkpls_mixed_example()


if __name__ == "__main__":
    TestMFKmixed().run_mfk_mixed_example()
    TestMFKmixed().run_mfkpls_mixed_example()
    unittest.main()
