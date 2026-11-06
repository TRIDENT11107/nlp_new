"""Trainable Keras time-series LSTM and dense reconstruction autoencoder."""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .preprocessing import monthly_financial_dataset, expense_windows, chronological_indices, autoencoder_features, autoencoder_train_count

tf.keras.utils.set_random_seed(27)

def expense_series(transactions):
    return monthly_financial_dataset(transactions)['expenses'].to_numpy(dtype=float)

def train_lstm(values, epochs=45):
    values=np.asarray(values,dtype=float)
    if len(values)<6: raise ValueError('At least six monthly observations are needed for an LSTM.')
    Xraw,yraw,w,target_indices=expense_windows(values)
    # Split target months in order; validation and test windows may use earlier
    # history as context, but never become training targets.
    n=len(yraw); splits=chronological_indices(n); train_end=splits['train_end']; val_end=splits['validation_end']
    scaler=MinMaxScaler().fit(yraw[:train_end].reshape(-1,1))
    X=np.stack([scaler.transform(row.reshape(-1,1)).ravel() for row in Xraw]); y=scaler.transform(yraw.reshape(-1,1)).ravel()
    Xtr,ytr=X[:train_end],y[:train_end]; Xv,yv=X[train_end:val_end],y[train_end:val_end]; Xt,yt=X[val_end:],y[val_end:]
    def build():
        m=tf.keras.Sequential([tf.keras.layers.Input((w,1)),tf.keras.layers.LSTM(64,return_sequences=True),tf.keras.layers.Dropout(.2),tf.keras.layers.LSTM(32),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
        m.compile(optimizer='adam',loss='mse'); return m
    model=build(); hist=model.fit(Xtr[...,None],ytr,validation_data=(Xv[...,None],yv),epochs=epochs,batch_size=min(8,len(Xtr)),verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)])
    # Refit on train + validation for the selected epoch count, then evaluate
    # exactly once on the chronologically held-out test targets.
    best_epoch=max(1,int(np.argmin(hist.history['val_loss'])+1)); final=build()
    final.fit(X[:val_end,...,None],y[:val_end],epochs=best_epoch,batch_size=min(8,val_end),verbose=0)
    model=final
    pred_scaled=model.predict(Xt[...,None],verbose=0).ravel()
    pred=scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel(); actual=yraw[val_end:]
    if not len(actual): actual=yraw[-len(pred):]
    future=[]; seq=list(scaler.transform(values.reshape(-1,1)).ravel()[-w:])
    for _ in range(3):
        nxt=float(model.predict(np.array(seq[-w:]).reshape(1,w,1),verbose=0)[0,0]); future.append(nxt); seq.append(nxt)
    f=scaler.inverse_transform(np.array(future).reshape(-1,1)).ravel()
    mae=float(mean_absolute_error(actual,pred)); rmse=float(np.sqrt(mean_squared_error(actual,pred))); valid=actual!=0; mape=float(np.mean(np.abs((actual[valid]-pred[valid])/actual[valid]))*100) if np.any(valid) else None
    baseline=scaler.inverse_transform(Xt[:,-1].reshape(-1,1)).ravel()
    baseline_metrics={'mae':float(mean_absolute_error(actual,baseline)),'rmse':float(np.sqrt(mean_squared_error(actual,baseline)))}
    baseline_metrics['mape']=float(np.mean(np.abs((actual[valid]-baseline[valid])/actual[valid]))*100) if np.any(valid) else None
    test_dates=target_indices[val_end:]
    return {'model':model,'history':hist.history,'window':w,'actual':actual,'predicted':pred,'future':f,'mae':mae,'rmse':rmse,'mape':mape,'baseline':baseline_metrics,'baseline_predicted':baseline,'scaler':scaler,'epochs':len(hist.history['loss']),'best_epoch':best_epoch,'batch_size':min(8,val_end),'train_samples':train_end,'validation_samples':val_end-train_end,'test_samples':n-val_end,'parameters':int(model.count_params()),'input_shape':(None,w,1),'test_indices':test_dates,'split':{'train':train_end,'validation':val_end-train_end,'test':n-val_end,'test_start_index':int(test_dates[0]) if len(test_dates) else None}}

def train_autoencoder(transactions, epochs=35):
    d,Xframe,scaler,feature_names=autoencoder_features(transactions)
    if len(d)<8: raise ValueError('At least eight valid expenses are needed for autoencoder training.')
    X=Xframe.to_numpy(dtype='float32'); train_end=autoencoder_train_count(len(X)); Xtr,Xv=X[:train_end],X[train_end:]
    inp=tf.keras.Input((X.shape[1],)); x=tf.keras.layers.Dense(8,activation='relu')(inp); x=tf.keras.layers.Dense(3,activation='relu')(x); x=tf.keras.layers.Dense(8,activation='relu')(x); out=tf.keras.layers.Dense(X.shape[1])(x)
    model=tf.keras.Model(inp,out); model.compile(optimizer='adam',loss='mse'); hist=model.fit(Xtr,Xtr,epochs=epochs,batch_size=min(16,len(Xtr)),validation_data=(Xv,Xv),verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)])
    errors=np.mean((X-model.predict(X,verbose=0))**2,axis=1); train_errors=errors[:train_end]; threshold=float(np.quantile(train_errors,.95))
    d['reconstruction_error']=errors; d['anomaly']=errors>threshold
    return {'model':model,'history':hist.history,'threshold':threshold,'data':d,'errors':errors,'scaler':scaler,'features':Xframe,'feature_names':feature_names,'train_samples':len(Xtr),'validation_samples':len(Xv),'threshold_quantile':.95,'threshold_source':'training reconstruction errors','threshold_training_errors':train_errors,'epochs':len(hist.history['loss']),'parameters':int(model.count_params()),'latent_dimension':3,'input_shape':(None,X.shape[1]),'batch_size':min(16,len(Xtr))}
