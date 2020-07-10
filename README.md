# Local-to-Global Support Vector Machines (LGSVMs)

**Authors**: F. Marchetti and E. Perracchione.

If you make use of this code, please cite [PAPER], which is the reference paper for further information concerning this classifier.

 <div class="row">
    <img src="https://iili.io/JyowwQ.png" width="270">
    <img src="https://iili.io/JyojAx.png" width="270">
    <img src="https://iili.io/JyoeoB.png" width="270">
  <caption> <p align="center"> LGSVMs: the grid approach. </p> </caption>
</div>
<br />

We propose a scikit-learn-based Python implementation of the LGSVM method. This classifier is constructed upon multiple local SVM models, that are then merged and glued together in order to obtain a global classifier. The presented setting is capable of speeding up standard SVMs, especially when treating a considerable amount of data.

The proposed LGSVM class estimator includes _fit_, _predict_ and _fit_predict_ methods. For more information concerning the parameters of the method please refer to the file *lgsvm.py* and to the reference paper. Moreover, in *demo_lgsvm.py* we propose a demo example.

<br />
 <div class="row">
    <img src="https://iili.io/JyoNtV.png" width="270">
    <img src="https://iili.io/JyoSKF.png" width="270">
    <img src="https://iili.io/JyovP1.png" width="270">
  <caption> <p align="center"> LGSVMs: the sparse approach. </p> </caption>
</div>




