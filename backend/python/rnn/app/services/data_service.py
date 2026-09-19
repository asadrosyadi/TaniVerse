import pandas as pd
import json
import requests
from datetime import datetime
import os

class DataService:
    """Manages plant data storage and retrieval (from file or URL)"""
    
    def __init__(self, history_source):
        self.history_source = history_source  # Bisa file path atau URL
        self.plant_data = {}
        self._load_history()
    
    def _load_history(self):
        """Load plant history from file or URL"""
        try:
            if self.history_source.startswith("http://") or self.history_source.startswith("https://"):
                response = requests.get(self.history_source)
                response.raise_for_status()
                raw_data = response.json()

                # Pastikan struktur data dari server sesuai
                if isinstance(raw_data, list):
                    # Kalau hanya data satu tanaman, pakai ID 1 secara default
                    self.plant_data[1] = pd.DataFrame(raw_data)
                elif isinstance(raw_data, dict):
                    for iot_id, readings in raw_data.items():
                        self.plant_data[int(iot_id)] = pd.DataFrame(readings)

                print(f"[DataService] Loaded history from URL for {len(self.plant_data)} plants")

            else:
                if os.path.exists(self.history_source):
                    with open(self.history_source, 'r') as f:
                        history_data = json.load(f)
                        for iot_id, readings in history_data.items():
                            self.plant_data[int(iot_id)] = pd.DataFrame(readings)
                    print(f"[DataService] Loaded history from file for {len(self.plant_data)} plants")

        except Exception as e:
            print(f"[DataService] Error loading history: {e}")
    
    def save_history(self):
        """Save plant history to file (skip if using URL)"""
        if self.history_source.startswith("http"):
            # Tidak menyimpan kalau data dari URL
            print("[DataService] Skipping save_history: source is URL")
            return

        try:
            history_data = {}
            for pid, pdata in self.plant_data.items():
                pdata_copy = pdata.copy()
                if 'created_at' in pdata_copy.columns:
                    pdata_copy['created_at'] = pdata_copy['created_at'].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                history_data[str(pid)] = pdata_copy.to_dict('records')

            with open(self.history_source, 'w') as f:
                json.dump(history_data, f)
            print(f"[DataService] History saved: {len(history_data)} plants")

        except Exception as e:
            print(f"[DataService] Error saving history: {e}")

    def add_sensor_reading(self, data):
        """Add sensor reading to plant history"""
        iot_id = data['iot_id']
        df_row = pd.DataFrame([data])

        # Simpan dalam memori
        if iot_id not in self.plant_data:
            self.plant_data[iot_id] = df_row
        else:
            self.plant_data[iot_id] = pd.concat([self.plant_data[iot_id], df_row], ignore_index=True)

        # Hapus data lama (lebih dari 30 hari)
        if 'created_at' in self.plant_data[iot_id].columns:
            self.plant_data[iot_id]['created_at'] = pd.to_datetime(self.plant_data[iot_id]['created_at'], errors='coerce')
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            self.plant_data[iot_id] = self.plant_data[iot_id][
                self.plant_data[iot_id]['created_at'] > cutoff
            ]

        # Simpan hanya jika bukan URL
        self.save_history()

        return iot_id
    
    def get_plant_data(self, iot_id):
        """Get data for a specific plant"""
        return self.plant_data.get(iot_id, None)
    
    def get_all_iot_ids(self):
        """Get list of all plant IDs"""
        return list(self.plant_data.keys())
