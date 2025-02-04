from LSTM_i_pomocne_funkcije_i_objekti import *
import os 
from pathlib import Path

#glavna funckija za treniranje i tesitranje

#trentuni direktorij za default
os.chdir(Path(__file__).parent)

#POSTAVKE
validation_data_pickle_path = 'validation_data.pickle'
train_data_pickle_path = 'train_data.pickle'    
test_data_pickle_path = 'test_data.pickle'    
#generirat STD i Mean ako ne postoj
if(os.path.isfile('train_mean_i_std.pickle')) == False:
    print('Vec postoji odredei Mean i Std')
    Srednja_Vrijednosti_I_Standardna_Devijacija(train_data_pickle_path)
mean_std_pickle_path = 'train_mean_i_std.pickle'

#Treniranje
TrainNetwork(train_data_pickle_path=train_data_pickle_path,
            validation_data_pickle_path=validation_data_pickle_path,
            mean_std_pickle_path=mean_std_pickle_path,
            nastavi_postojecu_trenirat = False, # !!
            broj_featura=15,
            lstm_hidden_num=1024, #---- Sta vece moze
            dropout_value=0.,  #bilo je 0.2 ili 0.15
            epoch_num=100, 
            velicina_batcha=2000, #---
            l2=1e-5,
            learning_rate = 0.0001,   # na .01 ili 0.001 --- !!
            max_grad_clip_value = 8.0,
            validacijsko_strpljenje = 15,
            number_of_lstm_layers = 3, # -----
            is_bidirectional = True, #------
            leraning_rate_decay = True, #------
            is_exponential_decay = True, #-----
            exponential_decay_gama = 0.95, #-----sta je manji od 1,brže se smanjuje learning rate
            is_cosine_annealing = False, #----- **
            cosine_anneal_min_value = 0.0000001, #----- **
            cosine_anneal_epoha_kada_restart = 260, #-------- **
            is_exponential_prvi_kad_su_oba = True, #------- **
            epoch_kada_switch_scheduler = 100, #-----**
            ) #----
