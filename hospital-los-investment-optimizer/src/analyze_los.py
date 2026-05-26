import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
np.random.seed(7)
n=180
df=pd.DataFrame({
 'hospital_id':range(1,n+1),'beds':np.random.randint(80,650,n),'nurses_per_100_beds':np.random.uniform(18,55,n),
 'scanner_hours_per_day':np.random.uniform(4,24,n),'avg_patient_age':np.random.uniform(35,72,n),'emergency_share':np.random.uniform(.15,.65,n)})
df['avg_los_days']=8.4-.07*df.scanner_hours_per_day-.035*df.nurses_per_100_beds+.018*df.avg_patient_age+2.2*df.emergency_share+np.random.normal(0,.55,n)
df.to_csv('data/hospital_los_sample.csv',index=False)
X=df[['scanner_hours_per_day','nurses_per_100_beds','avg_patient_age','emergency_share','beds']]; y=df['avg_los_days']
model=LinearRegression().fit(X,y); pred=model.predict(X)
coef=pd.DataFrame({'feature':X.columns,'coefficient':model.coef_}).sort_values('coefficient')
coef.to_csv('outputs/investment_impact_ranking.csv',index=False)
summary=f"R2={r2_score(y,pred):.3f}
MAE={mean_absolute_error(y,pred):.2f} days
Top reducible lever: {coef.iloc[0].feature}
"
open('outputs/executive_summary.txt','w').write(summary)
plt.figure(); plt.scatter(df.scanner_hours_per_day,y); plt.xlabel('Scanner hours/day'); plt.ylabel('Average length of stay'); plt.title('Scanner availability vs hospital length of stay'); plt.savefig('outputs/scanner_vs_los.png',bbox_inches='tight')
print(summary)
