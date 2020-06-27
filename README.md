# Local-to-Global Support Vector Machines (LGSVMs)

**Authors**: F. Marchetti and E. Perracchione.

If you make use of this code and for further information concerning this classifier, please cite the reference paper [PAPER].

We propose a scikit-learn-based Python implementation of the LGSVM method. This classifier is constructed upon multiple local SVM models, that are then merged and glued together in order to obtain a global classifier. The presented setting is capable of speeding up standard SVMs, especially when treating a considerable amount of data.

The proposed LGSVM class estimator includes _fit_, _predict_ and _fit_predict_ methods. For more information concerning the parameters of the method please refer to the file *lgsvm.py* and to the reference paper. Moreover, in *script_lgsvm.py* we propose a demo example.







