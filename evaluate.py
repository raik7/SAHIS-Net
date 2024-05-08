from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras import backend as K

def evaluateModel(model, X_test, Y_test, batchSize, iters, results_save_path):

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = yp[1]
    yp = np.round(yp,0)
    yp = yp.ravel().astype(int)

    Y_test = np.round(Y_test,0)
    Y_test = Y_test.ravel().astype(int)

    intersection = yp * Y_test
    union = yp + Y_test - intersection

    jaccard = (np.sum(intersection)/np.sum(union))  

    dice = (2. * np.sum(intersection) ) / (np.sum(yp) + np.sum(Y_test))


    # c_matrix = confusion_matrix(Y_test, yp)
    # tn, fp, fn, tp = c_matrix.ravel()
    
    # jaccard = dice = ACC = SE = SP = PRE = 0.0

    # SE = tp/(tp+fn+1)
    # SP = tn/(tn+fp+1)
    # PRE = tp/(tp+fp+1)
    # ACC = (tn + tp) / (tn + fp + fn + tp)
    # jaccard = tp/(tp+fn+fp)
    # dice = 2*tp/(2*tp+fn+fp)

    print('Jaccard Index : '+str(jaccard))
    print('Dice Coefficient : '+str(dice))
    # print('PRE Coefficient : '+str(PRE))
    # print('Accuracy : '+str(ACC))
    # print('SE : '+str(SE))
    # print('SP : '+str(SP))

    fp = open(results_save_path + '\\jaccard-{}.txt'.format(iters),'a')
    fp.write(str(jaccard)+'\n')
    fp.close()
    fp = open(results_save_path + '\\dice-{}.txt'.format(iters),'a')
    fp.write(str(dice)+'\n')
    fp.close()


    fp = open(results_save_path + '\\best-jaccard-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(jaccard>float(best)):
        print('***********************************************')
        print('Jaccard Index improved from '+str(best)+' to '+str(jaccard))
        fp = open(results_save_path + '\\best-jaccard-{}.txt'.format(iters),'w')
        fp.write(str(jaccard))
        fp.close()
        model.save_weights(results_save_path + '\\modelW-{}.h5'.format(str(iters)))


    fp = open(results_save_path + '\\best-dice-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(dice>float(best)):
        print('***********************************************')
        print('Dice Index improved from '+str(best)+' to '+str(dice))
        fp = open(results_save_path + '\\best-dice-{}.txt'.format(iters),'w')
        fp.write(str(dice))
        fp.close()
