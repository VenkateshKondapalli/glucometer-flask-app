import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout # type: ignore
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler
import io


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs): # <-- Add **kwargs
        super(MultiHeadSelfAttention, self).__init__(**kwargs) # <-- Pass **kwargs
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs): # <-- Add **kwargs
        super(TransformerBlock, self).__init__(**kwargs) # <-- Pass **kwargs
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs): # <-- Add **kwargs
        super(TransformerEncoder, self).__init__(**kwargs) # <-- Pass **kwargs
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return x

class PositionalEncoding(Layer):
    def __init__(self, position, embed_dim, **kwargs): # <-- Add **kwargs
        super(PositionalEncoding, self).__init__(**kwargs) # <-- Pass **kwargs
        self.pos_encoding = self.positional_encoding(position, embed_dim)

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def positional_encoding(self, position, embed_dim):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(embed_dim)[np.newaxis, :],
                                     embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

custom_objects = {
    "MultiHeadSelfAttention": MultiHeadSelfAttention,
    "TransformerBlock": TransformerBlock,
    "TransformerEncoder": TransformerEncoder,
    "PositionalEncoding": PositionalEncoding,
}

model = None
try:
    model = tf.keras.models.load_model(
        'glucometer_app/model/glucometer_transformer_model.h5', 
        custom_objects=custom_objects,
        compile=False
    )
    model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

scaler = MinMaxScaler()

def process_xml_data(uploaded_file):
    """
    Parses the uploaded XML file and prepares the data frame.
    This is the same logic from your notebook.
    """
    # Using io.BytesIO to treat the uploaded file content as a file
    xml_content = uploaded_file.read()
    tree = ET.parse(io.BytesIO(xml_content))
    root = tree.getroot()
    data_list = []

    for parent in root.findall('./*'):
        event_type = parent.tag
        for event in parent.findall('.//event'):
            data_list.append({
                'timestamp': event.get('ts'),
                'event_type': event_type,
                'value': event.get('value')
            })

    df = pd.DataFrame(data_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    df_pivot = df.pivot_table(index='timestamp', columns='event_type', values='value', aggfunc='first')
    df_pivot.rename(columns={'glucose_level': 'glucose', 'meal': 'carbs', 'bolus': 'insulin'}, inplace=True)
    
    for col in ['glucose', 'carbs', 'insulin']:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan
            
    df_final = df_pivot[['glucose', 'carbs', 'insulin']].resample('5min').mean()
    df_final['glucose'].interpolate(method='linear', inplace=True)
    df_final['carbs'].fillna(0, inplace=True)
    df_final['insulin'].fillna(0, inplace=True)
    df_final.dropna(subset=['glucose'], inplace=True)
    
    return df_final



def make_prediction(df_processed):
    """
    Scales the data, makes a prediction, and inverse-transforms the result.
    """
    # Fit the scaler on the uploaded data and transform it
    df_scaled_values = scaler.fit_transform(df_processed)
    df_scaled = pd.DataFrame(df_scaled_values, index=df_processed.index, columns=df_processed.columns)

    # Get the last 24 time steps (2 hours) for prediction input
    last_window = df_scaled[['glucose', 'carbs', 'insulin']].values[-24:]
    
    # The model expects a batch of data, so add a dimension
    input_for_pred = np.expand_dims(last_window, axis=0)
    
    # Make the prediction
    prediction_scaled = model.predict(input_for_pred)
    
    # Inverse-transform the prediction to get the real glucose value (mg/dL)
    # Create a dummy array to match the scaler's 3-feature shape
    dummy_array = np.zeros((prediction_scaled.shape[1], 3))
    dummy_array[:, 0] = prediction_scaled.flatten()
    prediction_inversed = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Get historical data for plotting
    historical_data = df_processed['glucose'].tail(24) # Last 2 hours of real data
    
    return prediction_inversed, historical_data
