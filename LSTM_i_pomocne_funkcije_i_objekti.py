import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np

#definicija LSTM klase
class LSTM_NN(torch.nn.Module):
    def __init__(self, input_features_num: int, LSTM_hidden_layers_num: int, 
                 Dropout: float, num_layers: int = 1, is_bidirectional : bool = False):
        super(LSTM_NN, self).__init__()
        #šrvi input je broj stupaca, drugi velicina LSTM hidden state-a, 
        #batch first za tensor shpae, i broj stacking celija je default 1
        self._LSTM = torch.nn.LSTM(input_features_num,
                                   LSTM_hidden_layers_num,
                                   batch_first=True,
                                   num_layers=num_layers,
                                   bidirectional=is_bidirectional)
        #ako je biderectional
        if is_bidirectional:
            self.output_size = 2 * LSTM_hidden_layers_num
        else:
            self.output_size = LSTM_hidden_layers_num
        
        # inicijlizacija tensorm lstm matrica weight i bias, xavier za weights, 0 za bias
        for names, params in self._LSTM.named_parameters():
            if 'weight_ih' in names or 'weight_hh' in names:
                torch.nn.init.xavier_uniform_(params.data)
            elif 'bias' in names:
                params.data.fill_(0)

        # lstm output layer se normalizira
        self.layer_norm = torch.nn.LayerNorm(self.output_size)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(Dropout)

        # Projekcijski layer,FCC (fully connected layer) za projekciju iz LSTM na zeljeni label shape
        self.output_layer = torch.nn.Linear(self.output_size, input_features_num)

    def forward(self, input_data_tensor: torch.Tensor):
        '''glavna metoda za forward step ucenja'''
        
        # ovde sad forward pass krozLSTM 
        lstm_output, _ = self._LSTM(input_data_tensor)

        # samo jedan output je na izlazu
        zadnji_timestep = lstm_output[:,0,:]
        zadnji_timestep = self.layer_norm(zadnji_timestep)
        zadnji_timestep = self.dropout(zadnji_timestep)

        # Output layer , triba bit (batch size, 1, broj featurea)
        predikcija_mreze = self.output_layer(zadnji_timestep)
        predikcija_mreze = predikcija_mreze.unsqueeze(1)
        
        #return
        return predikcija_mreze

#normalizacijske vrijednosti
def Srednja_Vrijednosti_I_Standardna_Devijacija(train_data_pickle_path:str):
    print('Određivanje Mean i Std od Train Data')
    with open(train_data_pickle_path, 'rb') as file:
        data = pickle.load(file)  
    #samo uzimamo train data  
    data=data['train']
    #maknut dimenziju 1
    data=data.squeeze(1)
    
    # indexi svih featura
    x_indexi=[0,2,4]
    y_indexi=[1,3,5]
    vx_indexi=[6,8,10]
    vy_indexi=[7,9,11]
    r_indexi=[12,13,14]

    #buffer za return
    dict_normalizacijskih_vrijednosti={}
    
    #sad svakli mean and std
    #x
    data_podskup = data[:,x_indexi]
    mean_x = data_podskup.mean()
    std_x= data_podskup.std()
    dict_normalizacijskih_vrijednosti['x_mean'] = mean_x
    dict_normalizacijskih_vrijednosti['x_std'] = std_x

    #y
    data_podskup = data[:,y_indexi]
    mean_y = data_podskup.mean()
    std_y = data_podskup.std()
    dict_normalizacijskih_vrijednosti['y_mean'] = mean_y
    dict_normalizacijskih_vrijednosti['y_std'] = std_y

    #vx
    data_podskup = data[:,vx_indexi]
    mean_vx = data_podskup.mean()
    std_vx= data_podskup.std()
    dict_normalizacijskih_vrijednosti['vx_mean'] = mean_vx
    dict_normalizacijskih_vrijednosti['vx_std'] = std_vx

    #vy
    data_podskup = data[:,vy_indexi]
    mean_vy = data_podskup.mean()
    std_vy = data_podskup.std()
    dict_normalizacijskih_vrijednosti['vy_mean'] = mean_vy
    dict_normalizacijskih_vrijednosti['vy_std'] = std_vy

    #r
    data_podskup = data[:,r_indexi]
    mean_r = data_podskup.mean()
    std_r = data_podskup.std()
    dict_normalizacijskih_vrijednosti['r_mean'] = mean_r
    dict_normalizacijskih_vrijednosti['r_std'] = std_r

    #skupit ih po indexima u torch array s 15 stupaca za broadcasting vektoriziranje
    mean_poredani_u_array = torch.tensor(\
                       [mean_x,mean_y,mean_x,mean_y,mean_x,mean_y,
                        mean_vx,mean_vy,mean_vx,mean_vy,mean_vx,mean_vy,
                        mean_r,mean_r,mean_r])
    std_poredani_u_array = torch.tensor(\
                       [std_x,std_y,std_x,std_y,std_x,std_y,
                        std_vx,std_vy,std_vx,std_vy,std_vx,std_vy,
                        std_r,std_r,std_r])
    dict_normalizacijskih_vrijednosti['mean_poredani_u_array'] = mean_poredani_u_array
    dict_normalizacijskih_vrijednosti['std_poredani_u_array'] = std_poredani_u_array

    #save i return dict
    with open('train_mean_i_std.pickle','wb') as write_file:
        pickle.dump(dict_normalizacijskih_vrijednosti,write_file)
    print('Done')
    return dict_normalizacijskih_vrijednosti


#dataloder koji normalizira s obzirom na train data
#mean i std treba prije izracunati
class MojDataloader(Dataset):
    #osnovna klasa za lodat iz pickle data, vrijedi i za Train,Validation i Test pickle
    #ocekuje već prije pokrenut SrednjaVrijednostiIStandardnaDevijacija() koji stvara pickle
    #inace ce error izbacit
    def __init__(self, pickle_file_path:str,tip_data:str,train_mean_std_pickle_path:str):
        #loadat data
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        

        #loadat dict svih mean i std
        with open(train_mean_std_pickle_path, 'rb') as file:
            self.mean_std_dict = pickle.load(file)

        #s obzirom na tip data izvuc korektne 
        if(tip_data == 'train'):
            self.input_data = data['train']
            self.labels = data['label']
            ########## BRISATI #############
            self.input_data = data['train'][:1000,:,:]
            self.labels=data['label'][:1000,:,:]
            self.input_data= self.input_data.repeat(10000,1,1)
            self.labels=self.labels.repeat(10000,1,1)
            ################################

        elif(tip_data == 'validate'):
            self.input_data = data['validate']
            self.labels = data['label']

        elif(tip_data == 'test'):
            self.input_data = data['test']
            self.labels = data['label']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        #ovdje se normalizira svaki input 
        normaliziran_input = (self.input_data[idx]-self.mean_std_dict['mean_poredani_u_array'])\
            /self.mean_std_dict['std_poredani_u_array']
        labels = self.labels[idx]
        return normaliziran_input, labels


#helepr za spremanje Loss rezultata
def spremanje_loss_train_validation(train_losses, validation_losses):
    #train
    train_losses_dataframe = pd.DataFrame(train_losses,columns=['losses'])
    csv_path_za_train_losseve=os.path.join(os.getcwd(),'Model/Train_Lossevi.csv')
    #provjera jeli postoji da se apenda , a ako ne naprave novi
    if not pd.io.common.file_exists(csv_path_za_train_losseve):
        train_losses_dataframe.to_csv(csv_path_za_train_losseve, index=False)
    else:
        train_losses_dataframe.to_csv(csv_path_za_train_losseve,mode='a',header=False,index=False)

    #validacije
    validation_losses_dataframe = pd.DataFrame(validation_losses,columns=['validation'])
    csv_path_za_validation_losseve=os.path.join(os.getcwd(),'Model/Validation_Lossevi.csv')
    if not pd.io.common.file_exists(csv_path_za_validation_losseve):
        validation_losses_dataframe.to_csv(csv_path_za_validation_losseve, index=False)
    else:
        validation_losses_dataframe.to_csv(csv_path_za_validation_losseve,mode='a',header=False,index=False)

#RMSE loss
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


#Glavna Funkcija za pozivanje za treniranje NN, također može nastaviti učenje
def TrainNetwork(train_data_pickle_path:str,
                 validation_data_pickle_path:str,
                 mean_std_pickle_path:str,
                 broj_featura:int,
                 lstm_hidden_num:int, 
                 dropout_value:int,
                 epoch_num:int, 
                 velicina_batcha:int,
                 loss_funkcija = RMSELoss,
                 optimizer = torch.optim.Adam, 
                 l2:float=1e-5,
                 learning_rate:float = 0.01,
                 max_grad_clip_value = 8.0,
                 validacijsko_strpljenje : int = 10,
                 nastavi_postojecu_trenirat : bool = False,
                 leraning_rate_decay:bool = False,
                 is_exponential_decay:bool = False,
                 exponential_decay_gama : float = 0.95,
                 is_cosine_annealing:bool =False,
                 cosine_anneal_min_value:float = 0.0000001,
                 cosine_anneal_epoha_kada_restart : int = 10,
                 is_exponential_prvi_kad_su_oba : bool = False,
                 epoch_kada_switch_scheduler : int =100,
                 number_of_lstm_layers: int = 1,
                 is_bidirectional : bool = False):

    #za svaki slucaj pribacit se na lokalni direktorij
    os.chdir(Path(__file__).parent)

    #prvo se prebacujemo na GPU, ako nema izbcujemo exception
    if (torch.cuda.is_available() == True):
        device= "cuda"
    else:
        raise Exception('Cuda not properly setup!!')
    
    #provjeri jeli ima folder u koji ce bit mreza i njen metadata, ako nema napravi ga
    if(os.path.isdir(os.path.join(os.getcwd(), "Model")) == False):
        os.mkdir(os.path.join(os.getcwd(), "Model"))

    #loadat vec ucenu ili napavit novu u Model folderu
    if(nastavi_postojecu_trenirat == True):
        print('Nastavljamo Učenje več postoječe: ')
        path_do_modela = os.path.join(os.getcwd(),'Model/ModelWithMetadata.pth')
        checkpoint = torch.load(path_do_modela)
        model = LSTM_NN(input_features_num = checkpoint['input_features_num'],
                        LSTM_hidden_layers_num= checkpoint['LSTM_hidden_layers_num'],
                        Dropout = checkpoint['dropout'],
                        num_layers=checkpoint['number_lstm_layers'],
                        is_bidirectional=checkpoint['is_bidirectional'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    else:
        print('CREATING novi LSTM Model')
        model = LSTM_NN(input_features_num = broj_featura,
                        LSTM_hidden_layers_num = lstm_hidden_num,
                        Dropout = dropout_value,
                        num_layers=number_of_lstm_layers,
                        is_bidirectional=is_bidirectional)
        model = model.to(device)
        print('done')
    
    #odabrani loss, defalut RMSE
    odabrani_loss = loss_funkcija().to(device)
    #odabrani optimizator, defualt ADAM
    odabrani_optimizator = \
        optimizer(model.parameters(),lr=learning_rate, weight_decay=l2)
    #scheduler
    if(leraning_rate_decay==True):
        #prvo ako je samo expoenetial decay
        if(is_exponential_decay == True and is_cosine_annealing == False):
            scheduler = ExponentialLR(odabrani_optimizator, 
                                      gamma=exponential_decay_gama)
        #sada ako je samo cosine anneal
        elif(is_exponential_decay == False and is_cosine_annealing == True):
            scheduler = CosineAnnealingLR(odabrani_optimizator,
                                          cosine_anneal_epoha_kada_restart,
                                          cosine_anneal_min_value)
        #a sad ako boje koristimo, pokrenit jednog prvo pa switchat drugog kasnije
        else:
            if(is_exponential_prvi_kad_su_oba == True):
                scheduler = ExponentialLR(odabrani_optimizator, 
                                          gamma=exponential_decay_gama)
                odabran_je_prvi = 'exponential'
            else:
                scheduler = CosineAnnealingLR(odabrani_optimizator,
                                          cosine_anneal_epoha_kada_restart,
                                          cosine_anneal_min_value)
                odabran_je_prvi = 'cosine'


    #sada kreiramo datalodere
    print('Stvaranje Dataloadera')
    #train dataloader
    train_dataset = MojDataloader(train_data_pickle_path, "train", mean_std_pickle_path)
    train_DataLoader = DataLoader(train_dataset, batch_size=velicina_batcha, shuffle=True)  
    #validation dataloader
    validation_dataset = MojDataloader(validation_data_pickle_path, "validate", mean_std_pickle_path)
    validation_DataLoader = DataLoader(validation_dataset, batch_size=velicina_batcha, shuffle=True)  
    print('done')

    #sada bufferi za pračenje učenja
    training_lossevi = []
    validacijski_losevi = []
    prethodni_validacijski_loss = float('inf')
    improvment_counter = 0

    #################################################################
    #Treniranje petlja
    print('Treniranje LSTM-a :')
    for epoha in range(epoch_num):
        print(f'==== EPOHA: {epoha} ====')
        #korigirat scheduler ako ga treba switchat
        if(is_exponential_decay == True and is_cosine_annealing == True):
            if(epoha == epoch_kada_switch_scheduler):
                if(odabran_je_prvi == 'exponential'):
                    scheduler = CosineAnnealingLR(odabrani_optimizator,
                                          cosine_anneal_epoha_kada_restart,
                                          cosine_anneal_min_value)
                elif(odabran_je_prvi == 'cosine'):
                    scheduler = ExponentialLR(odabrani_optimizator, 
                            gamma=exponential_decay_gama)
        #swicth u train mode
        model.train()
        ukupni_training_loss = 0
        counter_za_pracenje_loss = 0
        #petlja po batchevima
        for inputs, labels in (train_DataLoader):
            #prebacit na GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            #prvo ocistit ram na nulu za gradiente
            odabrani_optimizator.zero_grad()
            #forward propagacija od NN
            outputs = model(inputs)
            #loss
            loss = odabrani_loss(outputs,labels)
            ukupni_training_loss += loss.item()
            #backward propagacija
            loss.backward()
            #klipanje gradijenta
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           max_grad_clip_value)
            #optimizer korak, tj. updejt of weights
            odabrani_optimizator.step()
            #pracenje loss u konzoli
            if(counter_za_pracenje_loss % 10 == 0):
                print(f'\r---> LOSS: {loss.item()} ',end='')
            counter_za_pracenje_loss += 1

        #srednja vrijednosti kroz cijelu epohu
        prosjecni_training_loss = ukupni_training_loss/len(train_DataLoader)
        training_lossevi.append(prosjecni_training_loss)
        
        #scheduler
        if(leraning_rate_decay == True):
            scheduler.step() 

        ##########################################################
        #validacijski mod
        model.eval()
        ukupni_validation_loss = 0
        with torch.no_grad():
            for inputs, labels in(validation_DataLoader):
                #prebacit na GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                #forward propagacija od NN
                outputs = model(inputs)
                #loss
                loss = odabrani_loss(outputs,labels)           
                ukupni_validation_loss += loss.item()
            #srednja vrijednosti kroz cijelu epohu
            prosjecni_validacijski_loss = ukupni_validation_loss/len(validation_DataLoader)
        
        #izvjesce nakon epohe
        print()
        print(f'==== Rezultati epohe {epoha} :Prosjecni Training LOSS --> {prosjecni_training_loss:.12f},Prosjecni Validacijski LOSS --> {prosjecni_validacijski_loss:.12f} ====')
        
        #korak za prvojeru napretka te zaustavljanje ako napretka nema
        if(torch.round(torch.tensor(prosjecni_validacijski_loss),decimals=4) >= torch.round(torch.tensor(prethodni_validacijski_loss),decimals=4)):
            improvment_counter+= 1
            if(improvment_counter >= validacijsko_strpljenje):
                print('Early Stop jer nije improvementa bilo')
                break
        else:
            prethodniValidacijskiLoss = prosjecni_validacijski_loss 
           
    
    ####################################
    #Konačno spremanje METADATA mreže
    print()
    print('Spremanje Modela, Model Metadata, train i validation loss')
    spremanje_loss_train_validation(train_losses=training_lossevi, 
                                    validation_losses=validacijski_losevi)
    torch.save({
            'input_features_num': broj_featura,
            'LSTM_hidden_layers_num': lstm_hidden_num,
            'dropout': dropout_value,
            'number_lstm_layers':number_of_lstm_layers,
            'is_bidirectional':is_bidirectional,
            'model_state_dict': model.state_dict()
            },os.path.join(os.getcwd(),'Model/model_i_metadata.pth'))
    print('done')


#funkcija za tesiranje mreže
def TestNetwork(test_data_pickle_path:str,
                mean_std_pickle_path:str,
                broj_featura:int, 
                velicina_batcha:int):
    #GPU
    if (torch.cuda.is_available() == True):
        device = "cuda"
    else:
        raise Exception('Cuda nije dobro postavljena')
    
    #load gotove mreže
    path_do_modela = os.path.join(os.getcwd(),'Model/ModelWithMetadata.pth')
    checkpoint = torch.load(path_do_modela)
    model = LSTM_NN(input_features_num = checkpoint['input_features_num'],
                        LSTM_hidden_layers_num= checkpoint['LSTM_hidden_layers_num'],
                        Dropout = checkpoint['dropout'],
                        num_layers=checkpoint['number_lstm_layers'],
                        is_bidirectional=checkpoint['is_bidirectional'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    #test dataloader
    test_dataset = MojDataloader(test_data_pickle_path, "test", mean_std_pickle_path)
    test_DataLoader = DataLoader(test_dataset, batch_size=velicina_batcha, shuffle=True)  
    
    #evaluacija
    model.eval()
    print()
    print("TESTIRANJE MREZE: ")
    labels_lista = []
    predikcije_lista = []
    with torch.no_grad():
        for inputs, labels in (test_DataLoader):
            #prebacit na GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            #forward propagacija od NN
            outputs = model(inputs)
            #spremit predikciju i label, pripremit ih za pandas
            #labels za append
            labels_formated = labels.to('cpu')
            labels_formated = labels_formated.numpy()
            labels_formated = labels_formated.reshape(-1,1)
            labels_lista.append(labels_formated)
            #predikcije za append
            predikcije_formated = outputs.to('cpu')
            predikcije_formated = predikcije_formated.numpy()
            predikcije_formated = predikcije_formated.reshape(-1,1)
            predikcije_lista.append(predikcije_formated)


    #spremanje .csv lanels-predikcije 
    labels = np.concatenate(labels_lista)
    output = np.concatenate(predikcije_lista)
    output = np.hstack((labels,output))
    output = pd.DataFrame(output, columns = ['LABELS', 'PREDIKCIJE'])
    path_za_spremanje = os.path.join(os.getcwd(),'Model/Rezultati.csv')
    output.to_csv(path_do_modela,columns=['LABELS','PREDIKCIJE'])
    print('Saved --> Rezultati.csv sa labels i predickijma u Model folder !!!')

