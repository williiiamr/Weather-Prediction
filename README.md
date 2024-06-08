# Weather Prediction ☁️
<div align="center">
  <img src="https://github.com/williiiamr/ASL_Recoginition/blob/master/img/ASL_cover.png" alt="creditcard">
</div>

## Project Overview 📑
Currently, the climate conditions in Indonesia are becoming difficult to predict. The main cause is believed to be due to the increase in global temperature, leading to climate change. This climate change has a negative impact on agricultural production. Unpredictable weather can potentially affect crop yields and the resources that need to be allocated. This impact is certainly not beneficial for PT. GGP, which heavily relies on crop yields to maintain their economic stability. Therefore, Machine Learning (ML) is needed to generate climate information as a basis for decision-making so that plant resource management can be more efficient and the risk of crop failure can be minimized.

## About The Dataset 📅
The project uses Data from PT. GGP BMKG Weather station with 18 features and 4 years worth of data starting from early 2020 until the end of 2024. Before pipelining the data into model for training we need to first do feature selection as to only use the most relevant features for the model and not all 18 features. the features that will be used in this modeling are AVG_TEMP, RAINFALL, and AVG_WINDSPEED. 
<div align="center">
  <img src="https://github.com/williiiamr/Weather-Prediction/blob/master/img/dataset.png" alt="creditcard" width='500'>
</div>

## Data Preprocessing 🔗
Before being ready for analysis, data must be preprocessed first. Data preprocessing refers to the processes of cleaning, normalization, splitting, and segmentation of data to improve its quality and make it suitable for analysis.

### **Cleaning**
In this section the data is inputed and then checked for any available missing value. if there are any missing value the model will use rolling mean to fill the value.
```
df = pd.read_excel('/content/cuaca_fix.xlsx', dtype={'Tanggal':str} , parse_dates=False)
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df.set_index('Tanggal', inplace=True)
columns = ['MAX_TEMP', 'MIN_TEMP', 'SUNLIGHT', 'AVG_HUM', 'MAX_WINDSPEED', 'WINDIR']
df = df.drop(columns = columns)

df.isna().sum()
df.fillna(round(df.rolling(window=7, min_periods=1).mean(), 2), inplace=True)
df.info()df.info()
```
### **Splitting**
The next step is to split the data into 2 parts Training and Testing.
```
length = round(len(df) * 0.8)
train = df[:length]
test = df[length:]

print(f'Length of Training Data: {len(train)}')
print(f'Length of Test Data: {len(test)}')
```
### **Normalization**
After the data is splitted. Using Minmax scaler the data is then normalized into ranges of 0-1 for faster training and eliminating biases.
```
scaler = MinMaxScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
```

### **Segmentation**
Finally the data is grouped to ensure it meets the model's required format.
```
length = 7
batch_size = 32
generator = TimeseriesGenerator(train_scaled, train_scaled,
                                length=length,
                                batch_size=batch_size,
                                shuffle=False)

validation_generator = TimeseriesGenerator(test_scaled,test_scaled,
                                           length=length, batch_size=16, shuffle=False)
```

## GRU 🤖
The next step is to create a Gated Recurrent Unit (GRU) model to train the weather data. In building the GRU model, tensorflow is utilized from the keras library.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(128, activation='relu', input_shape=(n_steps,n_features), return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_features)
])
```

## Results ⭐
After the model is finished training, the model is then used to predict test data to see how robust it is in learning the data. This process also allow us to asses its capability before using it in real world practices. 
### AVG TEMP
<div align="center">
  <img src="https://github.com/williiiamr/Weather-Prediction/blob/master/img/AVG_TEMP.png" alt="AVG TEMP", width='400'>
</div>

### RAINFALL
<div align="center">
  <img src="https://github.com/williiiamr/Weather-Prediction/blob/master/img/RAINFALL.png" alt="RAINFALL", width='400'>
</div>

### AVG WINDSPEED
<div align="center">
  <img src="https://github.com/williiiamr/Weather-Prediction/blob/master/img/WINDSPEED.png" alt="AVG WINDSPEED", width='400'>
</div>

From the graph above the model is able to predict correctly most of the time meaning the model is able to learn well and are ready to be used in real world applications.

## Conclusion 💾
This project showcases the potential uses of machine learning in the realm of Weather Prediction. The developed model emerges as a potent asset, poised to empower companies in their endeavors to forecast weather patterns. By leveraging advanced machine learning techniques, this model furnishes decision-makers with invaluable insights, facilitating informed decision-making processes rooted in data-driven forecasts. While the model undoubtedly represents a significant stride forward, it is imperative to acknowledge its inherent imperfections and limitations. The writer harbors a fervent hope that through continued refinement and iterative improvement, this model will evolve into a more robust and reliable tool, further enhancing its utility and efficacy in addressing the intricate challenges of weather prediction.

## Suggestions 📎
- Experiment with other algorithms to find the most suitable for the available data.
- Collect more data to improve model performance.
- Develop a user friendly GUI to forecast next day weather using streamlit.

## Contributions 👨‍🔧
Contributions are welcome! Please submit a pull request if you have any suggestions or improvements.
