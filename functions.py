# My functions and classes

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def stratifyC1D(datainput, stratFeat=None ,binmethod='LinearBins', nbins=7, logtransform=False, adjust=False, graphs=True):

    data = datainput.copy()

    if stratFeat:

        #Perform log-transform
        if logtransform:

            data['log_'+stratFeat] = np.log(1+data[stratFeat])
            stratFeat = 'log_' + stratFeat


        #Apply chogen binning method for defining categories
        if binmethod=='LinearBins':

            cats = np.linspace(min(data[stratFeat]),
                               max(data[stratFeat]),
                               nbins)


        if binmethod=='QuantileBins':
            
            quantiles = np.linspace(0,100,nbins)
            cats = np.percentile(data[stratFeat],quantiles) 


        #Classificating each instance - stratfeat min distance from cats
        data[stratFeat+'_cat'] = np.argmin(abs(np.repeat(cats,len(data)).reshape((len(cats),len(data))) - data[stratFeat].values),0)


        #Agrupa as duas primeiras e as duas categoria
        data[stratFeat+'_adjust_cat'] = data[stratFeat+'_cat'].replace(0,1).replace(nbins-1,nbins-2)

        #Plotting
        if graphs:

            plt.figure(figsize=(10,5))
            plt.subplot(1,3,1)
            data[stratFeat].hist()
            for cat in cats:
                plt.axvline(cat,color='k',linestyle='--')
            plt.title('Original Distribution')

            plt.subplot(1,3,2)
            data[stratFeat+'_cat'].hist()
            plt.title('Categorical Distribution')

            plt.subplot(1,3,3)
            data[stratFeat+'_adjust_cat'].hist()
            plt.title('Categorical Distribution Ajusted')


        #Choosing ajusted
        if adjust:
            
            data = data.drop(axis=1,labels=[stratFeat+'_cat']).rename(columns={stratFeat+'_adjust_cat':stratFeat+'_cat'})
            cats = cats[1:-1]
       
        else:

            data = data.drop(axis=1,labels=[stratFeat+'_adjust_cat'])

    
        return data, cats

    
    else:
        print('Input feature to use as base for stratification.')




def create_sets(datainput,stratFeat=None):

    from sklearn.model_selection import StratifiedShuffleSplit

    if stratFeat:

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(datainput, datainput[stratFeat+'_cat']):
            strat_train_set = datainput.loc[train_index]
            strat_test_set = datainput.loc[test_index]

    else:

        print('Input feature to use as base for stratification.')

    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(datainput, test_size=0.2, random_state=42)

    return train_set, test_set, strat_train_set, strat_test_set


def calculate_errors(datainput, train_set, test_set, strat_train_set, strat_test_set, stratFeat):

    propDict = {}

    propDict['datainput'] = datainput[stratFeat+'_cat'].value_counts() / len(datainput)
    propDict['train'] = train_set[stratFeat+'_cat'].value_counts() / len(train_set)
    propDict['test'] = test_set[stratFeat+'_cat'].value_counts() / len(test_set)
    propDict['strat_train'] = strat_train_set[stratFeat+'_cat'].value_counts() / len(strat_train_set)
    propDict['strat_test'] = strat_test_set[stratFeat+'_cat'].value_counts() / len(strat_test_set)


    propDF = pd.DataFrame(propDict)

    propDF['train_test_err'] = np.abs(propDict['train'] - propDict['test'])/(propDict['train']+propDict['test'])*2
    propDF['strat_train_test_err'] = np.abs(propDict['strat_train'] - propDict['strat_test'])/(propDict['strat_train']+propDict['strat_test'])*2
    propDF['train_datainput_err'] = np.abs(propDict['train'] - propDict['datainput'])/propDict['datainput']
    propDF['strat_train_datainput_err'] = np.abs(propDict['strat_train'] - propDict['datainput'])/propDict['datainput']
    propDF['test_datainput_err'] = np.abs(propDict['test'] - propDict['datainput'])/propDict['datainput']
    propDF['strat_test_datainput_err'] = np.abs(propDict['strat_test'] - propDict['datainput'])/propDict['datainput']

    propDF = propDF.sort_index()

    return propDF

def plot_errors(propDF): 

    fig, ax1 = plt.subplots(figsize=(10,4))

    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Participation in each category')
    ax1.bar(propDF.index,propDF.datainput, alpha=0.3)
    #ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Relative error in participation')  # we already handled the x-label with ax1
    ax2.plot(propDF.train_test_err,label='train_test_err')
    ax2.plot(propDF.strat_train_test_err,label='strat_train_test_err')
    ax2.plot(propDF.train_datainput_err,label='train_datainput_err')
    ax2.plot(propDF.strat_train_datainput_err,label='strat_train_datainput_err')
    ax2.plot(propDF.test_datainput_err,label='test_datainput_err')
    ax2.plot(propDF.strat_test_datainput_err,label='strat_test_datainput_err')
    plt.legend(bbox_to_anchor=(1.08, 1), loc='upper left')
    plt.title('Impact of stratifying')
    #ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()