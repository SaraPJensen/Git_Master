from transformer import *


class smallDataset(data.Dataset):
    def __init__(self, seq_len, sample_size = 1000):
        super().__init__()

        self.max_start = 20

        self.seq_len = seq_len

        self.num_categories = self.max_start * 0.5**(seq_len-1)

        self.sample_size = sample_size

        array = np.empty((sample_size, seq_len))

        #self.num_categories = 100


        for i in range (sample_size):
            start = np.random.randint(0, self.max_start)

            for j in range(seq_len):
                array[i, j] = start

                start *= 2

        self.data = torch.from_numpy(array).long()



    def __len__(self):
        return self.data.size()[0]
        #return self.sample_size


    def __getitem__(self, idx):
        '''
        inp_data = self.data[idx]
                        
        # predict the next number (adding 1)
        change = inp_data - np.roll(inp_data,1 )
        change[0] = 0 
        labels = inp_data + change
        
        return inp_data, labels

        '''
        inp_data = self.data[idx]
        labels = self.data[idx]    #Or should something be changed?? 

        #What should be returned here??

        return inp_data, inp_data
        


#Create dataset and dataloaders
seq_len = 10
sample_size = 1000
full_dataset = smallDataset(seq_len, sample_size)


train_size = int(0.7 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset)-train_size -val_size
batch = 32
print("Size of dataloaders; train, val and test: ", train_size, val_size, test_size)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size,val_size, test_size])

train_loader = data.DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True, pin_memory=True)
val_loader   = data.DataLoader(val_dataset, batch_size=batch)
test_loader  = data.DataLoader(test_dataset, batch_size=batch)



##Training 

## Model params
input_dim =  20 * 2**(seq_len-1)   #full_dataset.num_categories*2   #Why?? 
model_dim = 32    ### increased from 32
num_heads = 1        ##increased from 1

num_classes = 20 * 2**(seq_len-1)   #What is this?? 

#num_classes=full_dataset.num_categories*2
num_layers=1      
dropout=0.0
lr=5e-4
warmup=50
max_epochs = 100
max_iters= max_epochs*len(train_loader)

## Training with validation
model = Transformer(input_dim, model_dim, num_classes, num_heads, num_layers,dropout)
model = model.to(device)

## Initializing optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# Apply lr scheduler per step
lr_scheduler = CosineWarmupScheduler(optimizer,
                                     warmup=warmup,
                                     max_iters=max_iters)






## Training with validation 

def calculate_loss(data, labels, model):
    seq_len = labels.size()[1]
    data = nnF.one_hot(data, num_classes = num_classes).float()
    target= model.forward(data, mask = subsequent_mask(data.size(-2),data.size(0)).to(device),add_positional_encoding=True)

    target_ = target.view(-1,target.size(-1))
    labels_ = labels.view(-1)
    
    loss = nnF.cross_entropy(target_, labels_ ) 
    
    return loss, target





def training(model, train_loader, val_loader,epochs,num_classes):
    
    # start with pretrained weights
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'saved_model_upd_dataset.pth')
    if os.path.isfile(pretrained_filename):
        checkpoint = torch.load(pretrained_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Early stopping
    patience = 10
    trigger_times = 0
    min_val_loss = np.inf
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        model.train()     
        for i,(data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data, labels = data.to(device), labels.to(device)
            loss,_ = calculate_loss(data, labels, model)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()    
        for data, labels in val_loader:
            if torch.cuda.is_available():
                data, labels = data.to(device), labels.to(device)
            loss,_ = calculate_loss(data, labels, model)
            # Calculate Loss
            val_loss += loss.item()

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {val_loss / len(val_loader)}')
        

        print('trigger times: 0')
        trigger_times = 0
        print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        min_val_loss = val_loss
        # Saving State Dict
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        pretrained_filename = os.path.join(CHECKPOINT_PATH,'saved_model_upd_dataset_final.pth')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, pretrained_filename)  
        
            
            
    return model





# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = os.path.join(CHECKPOINT_PATH, 'saved_model_upd_dataset_final.pth')
if os.path.isfile(pretrained_filename):
    print("Found pretrained model, loading...")
    checkpoint = torch.load(pretrained_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
else:
    model = training(model, train_loader, val_loader,max_epochs,num_classes)




def test(model, test_loader):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for i,(data, labels) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                data, labels = data.to(device), labels.to(device)        
            loss, target = calculate_loss(data, labels, model)
            test_loss += loss.item()

            acc = (target.argmax(dim=-1) == labels).float().mean()
            test_acc += acc

        print("Loss:", test_loss / len(test_loader),"\n","Accuracy:", 100*(test_acc.item() / len(test_loader)))


test(model, test_loader)


data_input_test, labels_test = next(iter(test_loader))
inp_data_test = nnF.one_hot(data_input_test[:,:], num_classes=num_classes).float()
inp_data_test = inp_data_test.to(device)

pred_output =model.forward(inp_data_test,  mask = subsequent_mask(inp_data_test.size(-2),inp_data_test.size(0)).to(device))
_, output_test = torch.max(pred_output, dim =-1)
#print("Output test: ", output_test)


index = np.arange(0,10)
print("Index:")
print(index, "\n")
print("Input:")
print(data_input_test[index,:])
print("\n Output:")
print(output_test[index,:] )


'''

def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data =self.log('train_loss', loss) 
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    
    for row in range(num_layers):
        for column in range(num_heads):
            #print(attn_maps[row][column])
            ax[row][column].imshow(attn_maps[row][column], origin='upper', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
            # Rotate the tick labels and set their alignment.
            plt.setp(ax[row][column].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
    fig.subplots_adjust(hspace=0.5)
    plt.show()




data_input, labels = next(iter(val_loader))
inp_data = nnF.one_hot(data_input, num_classes=num_classes).float()
inp_data = inp_data.to(device)
attention_maps = model.get_attention_maps(inp_data,mask = subsequent_mask(inp_data.size(-2),inp_data.size(0)).to(device))
norm_maps = model.get_norm_maps(inp_data,mask = subsequent_mask(inp_data.size(-2),inp_data.size(0)).to(device))




print(attention_maps[0].shape)
print(norm_maps[0].shape)


plot_attention_maps(data_input, attention_maps, idx=0)


'''