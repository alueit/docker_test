import matplotlib.pyplot as plt
import pandas as pd
import json

from catboost import CatBoostClassifier

feature_select = ['сумма', 'доход', 'частота_пополнения',
                  'сегмент_arpu', 'частота', 'объем_данных',
                  'on_net', 'продукт_1', 'продукт_2',
                  'секретный_скор', 'income_freq_mul', 'sum_repl_freq_mul']

model = CatBoostClassifier()
model.load_model('./models/model_cb.cbm')




def make_pred(dt, path_to_file, path_to_figs):
    print('Importing pretrained model...')
    preds_proba = model.predict_proba(dt)[:, 1]
    preds = (preds_proba > 0.5) * 1

    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': preds
    })
    
    print('Prediction complete!')

    plot_vals = plt.hist(preds_proba)
    plt.title("Распределение предсказаний")

    plt.text(0, plot_vals[0][0], s=f"{plot_vals[0][0]:0.0f}")
    plt.text(1, plot_vals[0][-1], s=f"{plot_vals[0][-1]:0.0f}")

    plt.savefig(path_to_figs.replace('csv', "png"))
    print('Graph completed!')

    feature_importance = zip(feature_select, model.get_feature_importance().round(1))
    feature_importance = dict(sorted(feature_importance, key=lambda kv: -kv[1])[:5])

    with open(path_to_figs.replace('csv', "json"), 'w+', encoding= 'utf-8') as f:
        json.dump(feature_importance, f)
    print('Feature importance obtained!')
    
    return submission




# feature_importance = zip(feature_select, model_cb.get_feature_importance().round(1))

# feature_importance = dict(sorted(feature_importance, key=lambda kv: -kv[1]))
