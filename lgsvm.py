import numpy as np
from numpy import random
from scipy.sparse import spdiags
from sklearn import svm, preprocessing
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from itertools import product
import sys
    
##############################################################################

class LGSVM(BaseEstimator,ClassifierMixin):
    
    def __init__(self, metric = 'euclidean', radius = None,\
                 approach = 'sparse', n_centers = None,\
                 sparse_mode = 'rand', sparse_equi = False,\
                 precomputed_centers = None, grid_par = 1.1,\
                 aux_method = True, aux_metric = 'euclidean',\
                 svm_model = svm.SVC(kernel='linear'), mem_save = False,\
                 prints = True, random_seed = 42):

        """
        
        Local-to-Global Support Vector Machines (LGSVMs)
        
        Authors: Marchetti F. & Perracchione E.
    
        Parameters
        ----------
        
        metric : considered metric for determining the subdomains. For the 
                 'grid' approach, the only valid metrics are 
                 ['euclidean','cosine']. For the 'sparse' approach, all the 
                 metrics valid for sklearn.metrics.pairwise_distances are
                 available.
        
        radius : positive scalar, radius of the balls (subdomains). The 
                 default value, set to None, changes then according to the\
                 chosen approach and metric. For the 'grid' approach in\
                 2D-3D, it is set in order to cover the domain depending on\
                 n_centers.
                 
        approach : chosen approach ['sparse','grid','precomputed'].
                   Choosing 'precomputed', an array of predetermined centers
                   is given via 
        
        n_centers : positive integer that rules the number of centers.
                    If the chosen approach is 'sparse', n_centers corresponds
                    to the number of total centers (it could be slightly
                    different due to the proportional distribution of the 
                    number n_centers among the classes).
                    If the chosen approach is 'grid' and the metric is 
                    'euclidean', n_centers corresponds to the number of
                    centers for each dimension, while if the metric is
                    'cosine' it is the number of centers sampled in each 
                    [0,pi] arc.
                    If the chosen approach is 'precomputed', this parameter
                    is ignored.
        
        sparse_mode : the way to obtain the centers in the sparse approach,
                      parameter ignored if the approach is not 'sparse'.                        
                      If 'rand', the centers are uniformly randomly sampled
                      in each class. If 'border', the elements of each class
                      that are closer to the other classes are selected.
        
        sparse_equi : parameter ignored if the approach is not 'sparse'.
                      If False, the centers are distributed among the classes
                      proportionally to their numerosity, if True the centers
                      are equally distributed among the classes.
                      
        precomputed_centers : parameter ignored if the approach is not 
                              'sparse'. The array of predetermined centers.
                              
        grid_par : parameter ignored if the approach is not 'grid'. Scalar
                   greater than 1, it rules the overlapping between the 
                   patches. It is preferable to set it in the interval (1,2).

        aux_method : if False, the auxiliary NC classifier is not employed
                     and test elements assigned to 'NoClass' might be
                     returned.
                     
        aux_metric : the metric adopted by the NC classifier.
        
        svm_model : the scikit-learn SVM model employed locally in each ball.
                    The decision_function_shape needs to be 'ovr'.
        
        mem_save : if True, big Nearest Neighbors arrays are avoided in order
                   in order to prevent memory issues, but the execution time
                   will be larger.
                   
        prints : if False, nothing will be displayed during training and
                 classification.
        
        random_state : positive integer. The seed the rules the 'rand' mode
                       in the 'sparse' approach.
        
        """
        
        self.metric = metric
        self.radius = radius
        self.approach = approach
        self.n_centers = n_centers
        self.sparse_mode = sparse_mode
        self.sparse_equi = sparse_equi
        self.precomputed_centers = precomputed_centers
        self.grid_par = grid_par
        self.aux_method = aux_method
        self.aux_metric = aux_metric
        self.svm_model = svm_model
        self.mem_save = mem_save
        self.prints = prints
        self.random_seed = random_seed

    def fit(self, X, y, return_centers = False):

        """
        
        LGSVM training
    
        Parameters
        ----------
        
        X : float numpy array (n_samples,n_features). The training set.
        
        y : numpy array of training labels (n_features,).
        
        return_centers : If True, the centers are returned in the output.
    
                              
        Returns
        -------
        
        ctrs : numpy array of centers if return_centers=True.
        
        """
        
        if self.prints == True:

            print("\n")
            print("LGSVM training")
            print("Mode:", self.approach)
            print("Metric:", self.metric)
        
        # Preprocessing and label encoding
        
        if self.metric == 'euclidean':
            the_scaler = preprocessing.MinMaxScaler()
        else:
            the_scaler = preprocessing.StandardScaler()
            
        self.scaler_fitted = the_scaler.fit(X)
        X = self.scaler_fitted.transform(X)
        
        le = preprocessing.LabelEncoder()
        self.le_fitted = le.fit(y)
        y = self.le_fitted.transform(y)    
        
        self.classes_num = len(np.unique(y))
        self.d = X.shape[1]
        
        # Definition of the centres
        
        if self.approach == 'precomputed':
            
            if self.precomputed_centers == None:
                sys.exit("Since approach is 'precomputed', precomputed_centers needs to be an array of centers of proper dimensions.")
                             
            if self.precomputed_centers.shape[1] != self.d:
                sys.exit("The number of features of the array of centers is not equal to the number of features of the training data.")
                             
            self.ctrs = self.scaler_fitted.transform(self.precomputed_centers)
        
        if self.approach == 'sparse':
                
            if self.n_centers == None:
                self.n_centers = int(np.around(X.shape[0]/10))
                
            self.ctrs = FindCtrs(X,y,self.classes_num,\
                            self.n_centers,self.metric,self.sparse_equi,\
                            self.sparse_mode,self.random_seed)
                
            if self.radius == None:
                self.radius = self.d/(X.shape[0]*self.ctrs.shape[0])
        
        if self.approach == 'grid':
            
            if self.metric != 'euclidean' and self.metric != 'cosine':
                sys.exit("'euclidean' and 'cosine' are the only metrics valid in the grid approach.")
                        
            if self.metric == 'euclidean':
                
                if self.n_centers == None:
                    self.n_centers = int(np.around((X.shape[0]/10)))
                
                self.ctrs = []
            
                for i in product(*tuple([np.linspace(0,1,self.n_centers)\
                                         for i in range(0,self.d)])):
                    self.ctrs.append(list(i))
                                        
                self.ctrs = np.array(self.ctrs)              
                
                if self.radius == None:
                    self.radius = self.grid_par*0.5*np.sqrt(self.d)\
                        /(self.n_centers-1)
            
            if self.metric == 'cosine':
                
                if self.n_centers == None:
                    self.n_centers = int(np.around((X.shape[0]/10)/self.d))
                
                self.ctrs = []
                
                for i in product(*tuple([np.linspace(0,np.pi,self.n_centers)\
                            for i in range(0,self.d-2)]+[np.linspace(0,\
                            2*np.pi,self.n_centers*2-1)[:-1]])):
                    self.ctrs.append([1]+list(i))
                            
                self.ctrs = np.unique(np.around(sphere2cart\
                                    (np.array(self.ctrs)),14),axis=0)
                    
                if self.radius == None:
                    
                    theta = np.pi/(self.n_centers-1)*0.5*self.grid_par
                    
                    if self.d == 2:
                        cos_ang = np.cos(theta)
                    else:
                        cos_ang = np.cos(theta)*((np.sin(theta))**2+(np.cos\
                                (theta))**3)+np.sin(theta)*(-np.sin(theta)*\
                                np.cos(theta)+np.sin(theta)*\
                                    (np.cos(theta))**2)
                    
                    self.radius = 1-cos_ang
        
        if self.prints == True:
            print("Number of centers:", self.ctrs.shape[0])
        
        # Definition of NN structure
        
        NN_struct = NearestNeighbors(radius = self.radius, metric = \
                                     self.metric).fit(X)
        
        if self.mem_save == False:
            NN_train = list(NN_struct.radius_neighbors(self.ctrs)[1])
            del NN_struct
        
        # Beginning of the training
        
        self.local_models = []  
        self.local_classes = []
        self.lonely_balls = [[] for i in range(0,self.classes_num)]
        
        # In lonely_balls, we store the indices of the balls characterized
        # by the presence of one class only

        for j in range(0,self.ctrs.shape[0]):
            
            if self.mem_save == False:
                idx = list(NN_train[j])
            else:
                idx = list(NN_struct.radius_neighbors(self.ctrs[j]\
                                                    .reshape(-1,1).T)[1])[0]

            ##################################
            
            if len(idx) == 0:
                
                self.local_models.append(None)
                self.local_classes.append(None)
            
            else:            
                
                y_loc = y[idx]
                
                if len(np.unique(y_loc))==1:

                    self.local_models.append(None)
                    self.local_classes.append(None)
                    self.lonely_balls[int(np.unique(y_loc)[0])]\
                        .append(j)
                else:
                    
                    X_loc = X[idx]
                    local_model = clone(self.svm_model) 
                    local_model.fit(X_loc,y_loc.ravel())
                    
                    self.local_models.append(local_model)
                    self.local_classes.append(np.unique(y_loc).tolist())
        
        if self.mem_save == False: 
            del NN_train
        
        if self.prints == True:
            print("Local models are ready!")  

        if self.aux_method == True:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.nc = NearestCentroid(metric=self.aux_metric).fit(X,y)
        
        if self.prints == True:            
            print("Auxiliary method is ready!")
            print("\n")
        
        if return_centers == True:
            return self.scaler_fitted.inverse_transform(self.ctrs)
        
    def predict(self, X):

        """
        
        LGSVM classification
    
        Parameters
        ----------
        
        X : float numpy array (n_samples_test,n_features). The test set.    
                              
        Returns
        -------
        
        y_predict : numpy array of predicted labels (n_samples_test,)
        
        """  
        
        if self.prints == True:
            print("LGSVM classification")
        
        # Preprocessing and initialization of the output array
        
        X = self.scaler_fitted.transform(X)
        y = np.zeros((X.shape[0],self.classes_num))

        # Definition of the support for the weight functions 
        
        wep = 1/self.radius  
        
        # Definition of the spatial matrix for weighting the local models
        
        with np.errstate(divide='ignore'):        
            SEM = spdiags((1/((wf(wep,pairwise_distances(X,self.ctrs,\
                metric=self.metric)))@np.ones((self.ctrs.shape[0],1)))).T,0,\
                X.shape[0],X.shape[0])@(wf(wep,pairwise_distances(X,\
                self.ctrs,metric=self.metric)))  
                                                                      
        # Definition of NN structure
        
        NN_struct = NearestNeighbors(radius = self.radius, metric = \
                             self.metric).fit(X)
            
        if self.mem_save == False:
            NN_test = list(NN_struct.radius_neighbors(self.ctrs)[1])
            del NN_struct

        for j in range(0,self.ctrs.shape[0]):

            if self.mem_save == False:           
                tidx = list(NN_test[j])
            else:
                tidx = list(NN_struct.radius_neighbors(self.ctrs[j]\
                                                     .reshape(-1,1).T)[1])[0]
            
            if len(tidx)>0:
                
                if self.local_models[j] != None:
                    X_loc = X[tidx]
                    
                    if len(self.local_classes[j]) == 2:
                        y_predict = self.local_models[j].decision_function\
                            (X_loc).reshape(-1,1)
                        y[tidx,self.local_classes[j][0]] += \
                            (-y_predict*np.asarray(SEM[tidx,j])).ravel()
                        y[tidx,self.local_classes[j][1]] += \
                                (+y_predict*np.asarray(SEM[tidx,j])).ravel()

                    else:
                        for i in range(0,len(self.local_classes[j])):
                            y_predict = self.local_models[j].\
                                decision_function(X_loc)[:,i].reshape(-1,1)
                            y[tidx,self.local_classes[j][i]] += \
                                (y_predict*np.asarray(SEM[tidx,j])).ravel()
        
        if self.mem_save == False: 
            del NN_test
        
        not_classified_local = list(np.where(~y.any(axis=1))[0])
        
        y = np.argmax(y,axis=1)
        
        not_classified = []
        
        if not_classified_local:
            
            if self.prints == True:
                print(len(not_classified_local),"elements out of",X.shape[0],\
                      "were not classified by local SVMs!")
                print("We look if they are in one-class balls...")
                
            NN_struct_ = NearestNeighbors(radius = self.radius, metric = \
                                 self.metric).fit(self.ctrs)
            
            if self.mem_save == False:
                
                NN_test_ = list(NN_struct_.radius_neighbors(\
                                        X[not_classified_local])[1])
            
                del NN_struct_
                
            for i in range(0,len(not_classified_local)):
                flag = 0
    
                for j in range(0,self.classes_num):
                    if self.mem_save == False:
                        local_nn = NN_test_[i]
                    else:
                        local_nn = list(NN_struct_.radius_neighbors(X\
                                    [not_classified_local][i].reshape(-1,1) 
                                    .T)[1])[0]
                            
                    if (len(local_nn)>0) and \
                        (set(local_nn)<=set(self.lonely_balls[j])):
                        y[int(not_classified_local[i])] = j
                        flag = 1
                        
                if flag == 0:
                    not_classified.append(int(not_classified_local[i])) 
                    
            if self.prints == True:                    
                print(len(not_classified_local)-len(not_classified),\
                      "elements were in one-class balls!")

        if not_classified:
            if self.aux_method == True:
                if self.prints == True:
                    print("The remaining",len(not_classified),\
                          "are classified by looking at the nearest \
                              centroid,")
                    print("They are the",np.around(len(not_classified)/len(y)\
                                            *100,2),"% of the total.\n")
                y[not_classified] = self.nc.predict(X[not_classified])
            
            else:
                if self.prints == True:
                    print('Not classified elements are assigned to no class')
                
        else:
            if self.prints == True:
                print("All the elements have been classified by local models!\n")
        
        ######################################################################
        
        if self.prints == True:
            print("Classification completed!")
            print("\n")
      
        # Decoding

        y = y.astype('int64')        
        y = self.le_fitted.inverse_transform(y.ravel())
        
        if not_classified and self.aux_method == False:
            y[not_classified] = 'NoClass'
            
        return y.ravel()

    def fit_predict(self, X, y, X_test, return_centers = False):
        
        """
        
        LGSVM training and classification
    
        Parameters
        ----------
        
        X : float numpy array (n_samples,n_features). The training set.
        
        y : numpy array of training labels (n_features,).

        X_test : float numpy array (n_samples_test,n_features). The test set. 
        
        return_centers : If True, the centers are returned in the output.
    
                              
        Returns
        -------
        
        y_predict : numpy array of predicted labels (n_samples_test,)
        
        ctrs : numpy array of centers if return_centers=True.
        
        """

        if self.prints == True:

            print("\n")
            print("LGSVM training and classification")
            print("Mode:", self.approach)
            print("Metric:", self.metric)
        
        # Preprocessing, label encoding and initialization of the output
        
        if self.metric == 'euclidean':
            the_scaler = preprocessing.MinMaxScaler()
        else:
            the_scaler = preprocessing.StandardScaler()
            
        self.scaler_fitted = the_scaler.fit(X)
        X = self.scaler_fitted.transform(X)
        
        le = preprocessing.LabelEncoder()
        self.le_fitted = le.fit(y)
        y = self.le_fitted.transform(y)    
        
        self.classes_num = len(np.unique(y))
        self.d = X.shape[1]
        
        X_test = self.scaler_fitted.transform(X_test)
        y_test = np.zeros((X_test.shape[0],self.classes_num))
        
        # Definition of the centres

        if self.approach == 'precomputed':
            
            if self.precomputed_centers == None:
                sys.exit("Since approach is 'precomputed', precomputed_centers needs to be an array of centers of proper dimensions.")
                             
            if self.precomputed_centers.shape[1] != self.d:
                sys.exit("The number of features of the array of centers is not equal to the number of features of the training data.")
                             
            self.ctrs = self.scaler_fitted.transform(self.precomputed_centers)
        
        if self.approach == 'sparse':
                
            if self.n_centers == None:
                self.n_centers = int(np.around(X.shape[0]/10))
                
            self.ctrs = FindCtrs(X,y,self.classes_num,\
                            self.n_centers,self.metric,self.sparse_equi,\
                            self.sparse_mode,self.random_seed)
                
            if self.radius == None:
                self.radius = self.d/(X.shape[0]*self.ctrs.shape[0])

        
        if self.approach == 'grid':

            if self.metric != 'euclidean' and self.metric != 'cosine':
                sys.exit("'euclidean' and 'cosine' are the only metrics valid in the grid approach.")
            
            if self.metric == 'euclidean':

                if self.n_centers == None:
                    self.n_centers = int(np.around((X.shape[0]/10)))
                
                self.ctrs = []
            
                for i in product(*tuple([np.linspace(0,1,self.n_centers)\
                                         for i in range(0,self.d)])):
                    self.ctrs.append(list(i))
                                        
                self.ctrs = np.array(self.ctrs)              
                
                if self.radius == None:
                    self.radius = self.grid_par*0.5*np.sqrt(self.d)\
                        /(self.n_centers-1)
            
            if self.metric == 'cosine':
                
                if self.n_centers == None:
                    self.n_centers = int(np.around((X.shape[0]/10)/self.d))
                
                self.ctrs = []
                
                for i in product(*tuple([np.linspace(0,np.pi,self.n_centers)\
                            for i in range(0,self.d-2)]+[np.linspace(0,\
                            2*np.pi,self.n_centers*2-1)[:-1]])):
                    self.ctrs.append([1]+list(i))
                            
                self.ctrs = np.unique(np.around(sphere2cart\
                                    (np.array(self.ctrs)),14),axis=0)
                    
                if self.radius == None:
                    
                    theta = np.pi/(self.n_centers-1)*0.5*self.grid_par
                    
                    if self.d == 2:
                        cos_ang = np.cos(theta)
                    else:
                        cos_ang = np.cos(theta)*((np.sin(theta))**2+(np.cos\
                                (theta))**3)+np.sin(theta)*(-np.sin(theta)*\
                                np.cos(theta)+np.sin(theta)*\
                                    (np.cos(theta))**2)
                    
                    self.radius = 1-cos_ang
        
        if self.prints == True:
            print("Number of centers:", self.ctrs.shape[0])
        
        # Definition of the NN training structure
        
        NN_struct = NearestNeighbors(radius = self.radius, metric = \
                                     self.metric).fit(X)
        if self.mem_save == False:
            
            NN_train = list(NN_struct.radius_neighbors(self.ctrs)[1])
            del NN_struct
        
        # Definition of the support for the weight functions 
        
        wep = 1/self.radius  

        # Definition of the spatial matrix for weighting the local models

        with np.errstate(divide='ignore'):        
            SEM = spdiags((1/((wf(wep,pairwise_distances(X_test,self.ctrs,\
                metric=self.metric)))@np.ones((self.ctrs.shape[0],1)))).T,0,\
                X_test.shape[0],X_test.shape[0])@(wf(wep,pairwise_distances\
                (X_test,self.ctrs,metric=self.metric)))  
                                                                      
        # Definition of the NN testing structure
        
        NN_struct__ = NearestNeighbors(radius = self.radius, metric = \
                             self.metric).fit(X_test)
        
        if self.mem_save == False:
            
            NN_test = list(NN_struct__.radius_neighbors(self.ctrs)[1])
            
            del NN_struct__
        
        lonely_balls = [[] for i in range(0,self.classes_num)]
        
        # In lonely_balls, we store the indices of the balls characterized
        # by the presence of one class only
        
        for j in range(0,self.ctrs.shape[0]):
            
            if self.mem_save == False:
                idx = list(NN_train[j])
            else:
                idx = list(NN_struct.radius_neighbors(self.ctrs[j].\
                                                      reshape(-1,1).T)[1])[0]

            if len(idx) != 0:

                y_loc = y[idx]
                
                if len(np.unique(y_loc))==1:

                    lonely_balls[int(np.unique(y_loc)[0])]\
                        .append(j)
                else:
                    X_loc = X[idx]
                    
                    if self.mem_save == False:           
                        tidx = list(NN_test[j])
                    else:
                        tidx = list(NN_struct__.radius_neighbors(self.ctrs[j]\
                                                    .reshape(-1,1).T)[1])[0]
        
                    if len(tidx)>0:
                        
                        local_model = clone(self.svm_model)
                    
                        local_model.fit(X_loc,y_loc.ravel())
                        local_classes = np.unique(y_loc).tolist()

                        X_loc = X_test[tidx]
                        
                        if len(local_classes) == 2:
                            y_predict = local_model.decision_function\
                                (X_loc).reshape(-1,1)
                            y_test[tidx,local_classes[0]] += \
                                (-y_predict*np.asarray(SEM[tidx,j])).ravel()
                            y_test[tidx,local_classes[1]] += \
                                    (+y_predict*np.asarray(SEM[tidx,j]))\
                                        .ravel()
    
                        else:
                            
                            for i in range(0,len(local_classes)):
                                y_predict = local_model.\
                                    decision_function(X_loc)[:,i].\
                                        reshape(-1,1)
                                y_test[tidx,local_classes[i]] += \
                                    (y_predict*np.asarray(SEM[tidx,j])).\
                                        ravel()                    
        if self.aux_method == True:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.nc = NearestCentroid(metric=self.aux_metric).fit(X,y)
                
        if self.prints == True:            
            print("Auxiliary method is ready!")
            print("\n")

        if self.mem_save == False: 
            del NN_test

        not_classified_local = list(np.where(~y_test.any(axis=1))[0])
        
        y_test = np.argmax(y_test,axis=1)
        
        not_classified = []
        
        if not_classified_local:
            
            NN_struct_ = NearestNeighbors(radius = self.radius, metric = \
                                 self.metric).fit(self.ctrs)
            if self.mem_save == False:
                
                NN_test_ = list(NN_struct_.radius_neighbors(X_test\
                                                    [not_classified_local])[1])
            
                del NN_struct_
            
            if self.prints == True:
                print(len(not_classified_local),"elements out of",X_test.\
                      shape[0],"were not classified by local SVMs!")
                print("We look if they are in one-class balls...")
                
            for i in range(0,len(not_classified_local)):
                flag = 0
                
                for j in range(0,self.classes_num):
                    
                    if self.mem_save == False:
                        local_nn = NN_test_[i]
                    else:
                        local_nn = list(NN_struct_.radius_neighbors(X_test\
                                    [not_classified_local][i].reshape(-1,1)\
                                    .T)[1])[0]
                    
                    if (len(local_nn)>0) and \
                        (set(local_nn)<=set(lonely_balls[j])):
                            
                        y_test[int(not_classified_local[i])] = j
                        flag = 1
                        
                if flag == 0:
                    not_classified.append(int(not_classified_local[i]))
                    
            if self.prints == True:                    
                print(len(not_classified_local)-len(not_classified),\
                      "elements were in one-class balls!")

        if not_classified:
            
            if self.aux_method == True:
                if self.prints == True:
                    print("The remaining",len(not_classified),\
                          "are classified by looking at the nearest centroid,")
                    print("They are the",np.around(len(not_classified)\
                                    /len(y_test)*100,2),"% of the total.\n")
                        
                y_test[not_classified] = self.nc.predict(X_test[not_classified])
            
            else:
                
                if self.prints == True:
                    print('Not classified elements are assigned to no class')
                
        else:
            
            if self.prints == True:
                print("All the elements have been classified by local models!\n")
        
        ######################################################################
        
        if self.prints == True:
            print("Classification completed!")
            print("\n")
      
        # Decoding

        y_test = y_test.astype('int64')        
        y_test = self.le_fitted.inverse_transform(y_test.ravel())
        
        if not_classified and self.aux_method == False:
            y_test[not_classified] = 'NoClass'
            
        if return_centers == True:
            return y_test.ravel(), self.scaler_fitted.inverse_transform\
                (self.ctrs)
            
        return y_test.ravel()        
        
##############################################################################
    
def wf(e,r):
    
    """
    
    Computes the Wendland's C2 function.

    """
    
    if type(e) is not float:
        e = e.T
        
    e = np.array(e)
    r = np.array(r)
    wf = np.matrix((np.multiply(np.power(np.fmax(1-(e*r),0*(e*r)),4),\
                                (4*(e*r)+1))))
    return wf

def sphere2cart(sphere):

    """
    
    Converts from n-spherical to cartesian coordinates.

    """
    
    cart = sphere[:,0].reshape(-1,1)*np.ones(sphere.shape)
    
    for i in range(0,sphere.shape[1]-1):
        ang_vec = np.hstack((np.cos(sphere[:,i+1]).reshape(-1,1),\
                              np.sin(sphere[:,i+1]).reshape(-1,1)*\
                              np.ones((sphere.shape[0],sphere.shape[1]-i-1))))
        cart[:,i:] = cart[:,i:]*ang_vec
    
    return cart

def FindCtrs(X,y,num_classes,n_centers,metric,equidistribution,mode,rand_seed):
    
    """
    
    Determines the centers in the 'sparse' approach.

    """    
    
    if mode == 'rand':
        
        if num_classes == 1:
            ctrs = FindCtrs_oneclass(X,n_centers,metric)
        
        else:
            
            ctrs = []
        
            for i in range(0,num_classes):
                X_oneclass = X[np.where(y==i)[0]]
                
                if equidistribution == True:
                    ctrs_local = FindCtrs_oneclass(X_oneclass,\
                                int(np.around(n_centers/num_classes)),metric,\
                                    rand_seed)
                else:    
                    ctrs_local = FindCtrs_oneclass(X_oneclass,\
                                int(np.around(n_centers*X_oneclass.shape[0]/\
                                    len(y))),metric,rand_seed)
                
                ctrs.append(ctrs_local)
       
    
    if mode == 'border':
        
        ctrs = []

        size = int(np.around(min(X.shape[0],10**(9)/(X.shape[0]*X.shape[1]))))
        
        if equidistribution == True:
            num = int(np.around(n_centers/num_classes))
                
        for i in range(0,num_classes):
            where_class = np.where(y==i)[0]
            
            num_cycles = int(np.ceil((len(y)-len(where_class))/size))
            
            D = np.min(pairwise_distances(X[where_class,:], \
                    X[np.where(y!=i)[0]][:int(size)],metric=metric),axis=1)
            
            for j in range(1,num_cycles):
                

                if j == num_cycles-1:
                    D = np.minimum(D,np.min(pairwise_distances(X\
                        [where_class], X[np.where(y!=i)[0]][int(j*size):],\
                        metric=metric),axis=1))
                else:                   
                
                    D = np.minimum(D,np.min(pairwise_distances(X[where_class],\
                        X[np.where(y!=i)[0]][int(j*size):int((j+1)*size)],\
                        metric=metric),axis=1))
                

            if equidistribution == False:
                num = int(np.around(n_centers*D.shape[0]/len(y)))

            indices = np.argsort(D)[:num]

            ctrs.append(X[where_class][indices])
            
    ctrs = [i for j in ctrs for i in j]
    
    return np.array(ctrs)


def FindCtrs_oneclass(X,n_centers,metric,rand_seed):
    
    """
    
    Finds the centers in a given class in the 'sparse' approach in 'rand' mode.

    """ 
    
    random.seed(rand_seed)
    indices = random.choice([i for i in range(0,X.shape[0])],size = n_centers,
                            replace = False)
    ctrs = X[indices,:]
    
    return ctrs