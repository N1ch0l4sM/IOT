from pymongo import MongoClient
import src.config as config

def init_collection():
    client = MongoClient(
        config.MONGO_CONFIG["host"],
        config.MONGO_CONFIG["port"],
        username=config.MONGO_CONFIG["username"],
        password=config.MONGO_CONFIG["password"],
    )
    db = client[config.MONGO_CONFIG["db"]]
    # Cria a coleção se não existir
    if "iot_weather" not in db.list_collection_names():
        db.create_collection("iot_weather")
        # Exemplo: criar índice
        # db.iot_weather.create_index("dt")
    print("Coleção iot_weather criada.")

if __name__ == "__main__":
    init_collection()