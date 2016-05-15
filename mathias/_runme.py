# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""
import os

PYBASH = "python "

os.system("{0}create_dataset_mm1.py".format(PYBASH))
os.system("{0}create_dataset_mm2.py".format(PYBASH))

os.system("{0}faron_xgb_01.py".format(PYBASH))
os.system("{0}faron_xgb_02.py".format(PYBASH))

os.system("{0}combine_oof_sets.py".format(PYBASH))