#!/usr/bin/env python3
from fastai.tabular import *

path = Config().data_path()/'rossmann'
train_df = pd.read_pickle(path/'train_clean')
train_df.head().T

n = len(train_df); n

idx = np.random.permutation(range(n))[:2000]
idx.sort()
small_train_df = train_df.iloc[idx[:1000]]
small_test_df = train_df.iloc[idx[1000:]]
small_cont_vars = ['CompetitionDistance', 'Mean_Humidity']
small_cat_vars = ['Store', 'DayOfWeek', 'PromoInterval']
small_train_df = small_train_df[small_cat_vars + small_cont_vars + ['Sales']]
small_test_df = small_test_df[small_cat_vars + small_cont_vars + ['Sales']]
small_train_df.head()
small_test_df.head()


categorify = Categorify(small_cat_vars, small_cont_vars)
categorify(small_train_df)
categorify(small_test_df, test=True)

small_test_df.head()
small_train_df.PromoInterval.cat.categories

small_train_df['PromoInterval'].cat.codes[:5]
fill_missing = FillMissing(small_cat_vars, small_cont_vars)
fill_missing(small_train_df)
fill_missing(small_test_df, test=True)
small_train_df[small_train_df['CompetitionDistance_na'] == True]

# Preparing the full data set
train_df = pd.read_pickle(path/'train_clean')
test_df = pd.read_pickle(path/'test_clean')
len(train_df), len(test_df)
procs = [FillMissing, Categorify, Normalize]
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen', 'Promo2Weeks',
            'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State',
            'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw', 'SchoolHoliday_fw', 'SchoolHoliday_bw']


cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity',
             'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend',
             'trend_DE', 'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']


dep_var = 'Sales'

df = train_df[cat_vars + cont_vars + [dep_var, 'Date']].copy()
test_df['Date'].min(), test_df['Date'].max()
cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()
cut

valid_idx = range(cut)
df[dep_var].head()
data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
        .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))
        .databunch())

max_log_y = np.log(np.max(train_df['Sales'])*1.2)
y_range = torch.tensor([0, max_log_y], device=defaults.device)

learn = tabular_learner(data, layers=[1000, 500], ps=[0.001, 0.01], emb_drop=0.04, y_range=y_range, metrics=exp_rmspe)
learn.model
len(data.train_ds.cont_names)
learn.lr_find()
learn.recorder.plot()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.save('1')
learn.recorder.plot_losses()
learn.load('1');
learn.fit_one_cycle(5, 3e-4, wd=0.2)
learn.save('2')
learn.load('2');
learn.fit_one_cycle(5, 3e-5, wd=0.2)
learn.save('3')
learn.load('3');
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, 7e-7, wd=0.2)
learn.fit_one_cycle(5, 1e-6, wd=0.2)



def get_lr():
    learn.lr_find()
    learn.recorder.plot(suggestion=True)


def show_dec(dec): print('{:10f}'.format(dec))


# Here's the stuffs for the second model
learn = tabular_learner(data, layers=[1000, 500], ps=[0.001, 0.01], emb_drop=0.05, y_range=y_range, metrics=exp_rmspe)
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_num_grad = 1.32e-02
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.save('big-1')
