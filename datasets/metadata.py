uci_datasets_info = {
    "factory": {
        "delim": ",",
        "size": 100000,
        "target_variable": [
            "machine_failure",
            "failure_type",
            "remaining_useful_life_hours",
            "machine_health_score", 
            "failure_probability", 
            "maintenance_priority"
        ],
        "categorical_variables": [
            "factory_id",
            "production_line",
            "machine_type",
            "manufacturer",
            "model",
            "factory_shift",
            "maintenance_type",
            "maintenance_priority",
            "failure_type",
            "machine_failure"
        ],
        "id_variables": [
            "machine_id",
            "monitoring_id",
            "timestamp"
        ]
    },
    "claims": {
        "delim": ",",
        "size": 678013,
        "target_variable": [
            "ClaimNb"
        ],
        "categorical_variables": [
            "Area", 
            "VehPower", 
            "VehBrand", 
            "VehGas", 
            "Region"
        ],
        "id_variables": [
            "IDpol"
        ]
    }
} 