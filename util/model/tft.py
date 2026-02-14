import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import torch

class TFTForecaster:
    def __init__(self, max_prediction_length=7, max_encoder_length=30, batch_size=32):
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.batch_size = batch_size
        self.tft = None
        
    def _prepare_dataset(self, df):
        # Add time index required by TFT
        df = df.copy()
        df['time_idx'] = (df['date'] - df['date'].min()).dt.days
        df['month'] = df['date'].dt.month.astype(str)
        
        return df

    def train(self, df):
        data = self._prepare_dataset(df)
        training_cutoff = data["time_idx"].max() - self.max_prediction_length

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="sales",
            group_ids=["store_id", "item_id"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["store_id", "item_id"],
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=["time_idx", "is_holiday", "is_promo"],
            time_varying_unknown_reals=["sales"],
            target_normalizer=GroupNormalizer(groups=["store_id", "item_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        train_dataloader = training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        self.tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=8,  # Kept small for demo speed
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
        )

        trainer = pl.Trainer(
            max_epochs=1,  # Set to 1 for quick demo testing
            accelerator='cpu', 
            enable_checkpointing=False,
            logger=False
        )
        
        print("Starting TFT training...")
        trainer.fit(self.tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        print("TFT Training Complete.")

    def predict(self, df):
        data = self._prepare_dataset(df)
        
        # Create prediction dataset covering the last window
        prediction_ds = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="sales",
            group_ids=["store_id", "item_id"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["store_id", "item_id"],
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=["time_idx", "is_holiday", "is_promo"],
            time_varying_unknown_reals=["sales"],
            target_normalizer=GroupNormalizer(groups=["store_id", "item_id"], transformation="softplus"),
            predict_mode=True
        )
        
        dataloader = prediction_ds.to_dataloader(train=False, batch_size=self.batch_size*10, num_workers=0)
        raw_predictions = self.tft.predict(dataloader, mode="prediction")
        return raw_predictions
