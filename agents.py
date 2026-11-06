"""Small, explainable financial agents operating on SQLite-backed frames."""
def spending_agent(transactions):
    d=transactions[transactions.kind=='Expense']
    totals=d.groupby('category').amount.sum().sort_values(ascending=False)
    return {'agent':'spending','largest_category':str(totals.index[0]) if len(totals) else None,'largest_total':float(totals.iloc[0]) if len(totals) else 0.0,'expense_total':float(d.amount.sum())}

def savings_agent(transactions, month):
    d=transactions[transactions.date.str.startswith(month)]
    inc=float(d.loc[d.kind=='Income','amount'].sum()); exp=float(d.loc[d.kind=='Expense','amount'].sum())
    return {'agent':'savings','income':inc,'expenses':exp,'net_savings':inc-exp,'savings_rate':(inc-exp)/inc if inc else 0.0}

def budget_agent(transactions, budgets, month):
    bd=budgets[budgets.month==month]; d=transactions[(transactions.kind=='Expense')&transactions.date.str.startswith(month)]
    actual=d.groupby('category').amount.sum(); rows=[]
    for _,r in bd.iterrows():
        spent=float(actual.get(r.category,0)); rows.append({'category':r.category,'budget':float(r.amount),'spent':spent,'utilization':spent/float(r.amount),'over':spent>r.amount})
    return {'agent':'budget','categories':rows,'overruns':[r for r in rows if r['over']]}

def portfolio_agent(investments):
    if investments.empty: return {'agent':'portfolio','value':0.0,'invested':0.0,'return_pct':0.0,'concentration':0.0,'largest_asset_class':None}
    d=investments.copy(); d['value']=d.quantity*d.current_price; d['invested']=d.quantity*d.purchase_price
    by_type=d.groupby('asset_type').value.sum(); value=float(d.value.sum()); invested=float(d.invested.sum())
    return {'agent':'portfolio','value':value,'invested':invested,'return_pct':(value/invested-1) if invested else 0.0,'concentration':float(by_type.max()/value) if value else 0.0,'largest_asset_class':str(by_type.idxmax()) if len(by_type) else None}

def health_agent(transactions, budgets, investments, monthly_income, reserve, month):
    sav=savings_agent(transactions,month); spend=spending_agent(transactions); bud=budget_agent(transactions,budgets,month); port=portfolio_agent(investments)
    income=sav['income'] or float(monthly_income or 0); rate=sav['net_savings']/income if income else 0
    expense_ratio=sav['expenses']/income if income else 1
    adherence=1-(len(bud['overruns'])/len(bud['categories']) if bud['categories'] else 0)
    reserve_score=min(1,float(reserve or 0)/max(income*3,1)); diversification=1-port['concentration']
    components={'savings':max(0,min(1,rate/.2))*30,'expense_burden':max(0,min(1,1-expense_ratio))*25,'budget_adherence':adherence*20,'emergency_reserve':reserve_score*15,'diversification':diversification*10}
    return {'agent':'health','score':round(sum(components.values())),'components':components,'savings':sav,'spending':spend,'budget':bud,'portfolio':port}

def market_agent(investments):
    """Summarize manually entered price marks without implying live market data."""
    p=portfolio_agent(investments)
    return {'agent':'market','tracked_value':p['value'],'tracked_return_pct':p['return_pct'],'source':'manually entered current prices','live_data':False}

def risk_agent(health):
    """Expose cash-flow and concentration indicators from local records."""
    s=health['savings']; p=health['portfolio']
    return {'agent':'risk','expense_to_income':s['expenses']/s['income'] if s['income'] else None,'asset_class_concentration':p['concentration'],'largest_asset_class':p['largest_asset_class'],'scope':'descriptive indicators; not a risk forecast'}

def learning_agent(lstm=None, autoencoder=None):
    return {'agent':'learning','lstm_available':lstm is not None,'lstm_test_mae':float(lstm['mae']) if lstm is not None else None,'lstm_baseline_mae':float(lstm['baseline']['mae']) if lstm is not None else None,'autoencoder_available':autoencoder is not None,'autoencoder_threshold':float(autoencoder['threshold']) if autoencoder is not None else None}

def notification_agent(health, autoencoder=None):
    notices=[{'type':'budget','category':r['category'],'utilization':r['utilization']} for r in health['budget']['overruns']]
    if autoencoder is not None:
        flagged=autoencoder['data'][autoencoder['data'].anomaly]
        notices.extend({'type':'unusual_transaction','category':str(r.category),'date':str(r.date),'amount':float(r.amount),'error':float(r.reconstruction_error)} for _,r in flagged.iterrows())
    return {'agent':'notification','items':notices}

def decision_coordinator(transactions, budgets, investments, profile, month, lstm=None, autoencoder=None):
    """Join measured agent outputs and optional trained model results."""
    result=health_agent(transactions,budgets,investments,profile.get('monthly_income',0),profile.get('emergency_savings',0),month)
    insights=[]
    if result['spending']['largest_category']:
        insights.append({'priority':'normal','source':'spending','text':f"Largest recorded spend is {result['spending']['largest_category']} ({result['spending']['largest_total']:.0f})."})
    for item in result['budget']['overruns']:
        insights.append({'priority':'high','source':'budget','text':f"{item['category']} is at {item['utilization']:.0%} of its budget."})
    if result['portfolio']['concentration']>.6:
        insights.append({'priority':'medium','source':'portfolio','text':f"{result['portfolio']['largest_asset_class']} represents {result['portfolio']['concentration']:.0%} of tracked portfolio value."})
    if lstm is not None:
        insights.append({'priority':'normal','source':'lstm','text':f"Held-out test MAE is {lstm['mae']:.0f}; next expense forecast is {lstm['future'][0]:.0f}."})
    if autoencoder is not None:
        flagged=int(autoencoder['data'].anomaly.sum())
        insights.append({'priority':'medium' if flagged else 'normal','source':'autoencoder','text':f"Autoencoder flagged {flagged} transactions above its reconstruction threshold."})
    insights=sorted(insights,key=lambda x:{'high':0,'medium':1,'normal':2}[x['priority']])
    evidence={
        'spending':{'supporting_data':f"{result['spending']['largest_category']}: {result['spending']['largest_total']:.2f} across recorded expenses",'reasoning':'This is the largest cumulative category in the local transaction ledger.','limitation':'Historical recorded spending only.'},
        'budget':{'supporting_data':'; '.join(f"{x['category']}: {x['spent']:.2f}/{x['budget']:.2f} ({x['utilization']:.0%})" for x in result['budget']['overruns']),'reasoning':'Recorded spending exceeds a saved category budget.','limitation':'Only configured budgets and recorded transactions are compared.'},
        'portfolio':{'supporting_data':f"{result['portfolio']['largest_asset_class']}: {result['portfolio']['concentration']:.0%} of tracked value",'reasoning':'Concentration is calculated from quantity multiplied by manually entered current prices.','limitation':'No live prices, holdings outside this ledger, or suitability information.'},
        'lstm':{'supporting_data':f"Test MAE {lstm['mae']:.2f}; naive MAE {lstm['baseline']['mae']:.2f}" if lstm is not None else '', 'reasoning':'Forecast comes from the trained LSTM and is evaluated against later held-out months.','limitation':'Short synthetic history; compare against the naive baseline.'},
        'autoencoder':{'supporting_data':f"{int(autoencoder['data'].anomaly.sum())} records exceed training threshold {autoencoder['threshold']:.5f}" if autoencoder is not None else '', 'reasoning':'Rows have larger reconstruction error than the training-derived threshold.','limitation':'Unusual does not mean fraudulent; inspect the transaction context.'}}
    for insight in insights:
        detail=evidence.get(insight['source'],{})
        insight.update(detail)
        insight['recommended_action']='Review the underlying records and decide whether follow-up is appropriate.'
    agents={name:{'agent':name,'status':'available'} for name in ('market','risk','portfolio','spending','savings','learning','explanation','notification')}
    agents.update({'market':market_agent(investments),'risk':risk_agent(result),'spending':result['spending'],'savings':result['savings'],'portfolio':result['portfolio'],'learning':learning_agent(lstm,autoencoder),'notification':notification_agent(result,autoencoder),'explanation':{'agent':'explanation','insights':insights}})
    return {'health':result,'insights':insights,'models':{'lstm':lstm is not None,'autoencoder':autoencoder is not None},'agents':agents}
