# IPM Attack Targeting Configuration Guide

## 🎯 Overview

Your IPM attack now supports **configurable targeting levels** for comprehensive cluster shuffling evaluation. You can easily switch between different targeting strategies by modifying the config.

## 📋 Configuration Options

### Target Levels (`target_level`)

#### 1. **Client-Level Targeting** (`"client"`)
```python
"target_level": "client"
```
- **Strategy:** Targets individual client's learning direction
- **Estimation:** Uses own gradient + history as benign estimate
- **Best for:** Testing basic attack effectiveness
- **Scale Factor:** Moderate (0.9x base)

#### 2. **Cluster-Level Targeting** (`"cluster"`) 
```python
"target_level": "cluster"
```
- **Strategy:** Targets cluster's collective learning direction  
- **Estimation:** Uses gradient history + cluster trends
- **Best for:** Testing cluster aggregation robustness
- **Scale Factor:** Strong (1.1x base)

#### 3. **Global-Level Targeting** (`"global"`)
```python
"target_level": "global"
```
- **Strategy:** Targets global model's expected learning direction
- **Estimation:** Uses long-term gradient history
- **Best for:** Testing overall system disruption
- **Scale Factor:** Maximum (1.2x base)

#### 4. **Adaptive Targeting** (`"adaptive"`)
```python
"target_level": "adaptive"
```
- **Strategy:** Automatically chooses best targeting method
- **Estimation:** Dynamic based on available context
- **Best for:** Sophisticated adaptive attacks
- **Scale Factor:** Confidence-based

## 🛠️ Example Configurations

### Configuration 1: Basic Client-Level Attack
```python
"ipm_config": {
    "manipulation_strategy": "maximize_distance",
    "target_level": "client",              # ← TARGET LEVEL
    "target_client": None,
    "aggregation_method": "fedavg",
    "lambda_param": 0.2,
    "cluster_awareness": False,            # Disable cluster features
    "adaptive_scaling": True
}
```

### Configuration 2: Cluster-Aware Attack  
```python
"ipm_config": {
    "manipulation_strategy": "maximize_distance", 
    "target_level": "cluster",             # ← TARGET LEVEL
    "target_client": None,
    "aggregation_method": "fedavg",
    "lambda_param": 0.2,
    "cluster_awareness": True,             # Enable cluster awareness
    "adaptive_scaling": True
}
```

### Configuration 3: Global Disruption Attack
```python
"ipm_config": {
    "manipulation_strategy": "maximize_distance",
    "target_level": "global",              # ← TARGET LEVEL  
    "target_client": None,
    "aggregation_method": "fedavg",
    "lambda_param": 0.2,
    "cluster_awareness": True,
    "adaptive_scaling": True
}
```

### Configuration 4: Smart Adaptive Attack
```python
"ipm_config": {
    "manipulation_strategy": "maximize_distance",
    "target_level": "adaptive",            # ← TARGET LEVEL
    "target_client": None,
    "aggregation_method": "fedavg", 
    "lambda_param": 0.2,
    "cluster_awareness": True,             # Enable all features
    "adaptive_scaling": True
}
```

## 🔬 Research Experiment Workflow

### Phase 1: Baseline Evaluation
1. **No Defense:** `clustering: False`, `target_level: "client"`
2. **Measure:** Attack effectiveness, model performance

### Phase 2: Client-Level vs Cluster Defense  
1. **Enable Clustering:** `clustering: True`
2. **Client Attack:** `target_level: "client"` 
3. **Cluster Attack:** `target_level: "cluster"`
4. **Compare:** Which is more/less effective with clustering?

### Phase 3: Sophisticated Attacks
1. **Global Attack:** `target_level: "global"`
2. **Adaptive Attack:** `target_level: "adaptive"`
3. **Analyze:** Can cluster shuffling stop advanced attacks?

### Phase 4: Defense Optimization
1. **Vary cluster sizes, reshuffling frequency**
2. **Test against all target levels**
3. **Find optimal cluster shuffling parameters**

## 🎯 Expected Results

| Target Level | vs No Defense | vs Cluster Shuffling | Research Insight |
|-------------|---------------|---------------------|------------------|
| `client`     | High effectiveness | Reduced effectiveness | Basic defense works |
| `cluster`    | High effectiveness | Moderate effectiveness | Some resistance |
| `global`     | Very high effectiveness | Low-moderate effectiveness | Strong defense needed |
| `adaptive`   | Highest effectiveness | Effectiveness varies | Ultimate test |

## 🚀 Quick Test Commands

```bash
# Test client-level attack
# Edit config.py: "target_level": "client" 
python3 main.py

# Test cluster-level attack  
# Edit config.py: "target_level": "cluster"
python3 main.py

# Test global-level attack
# Edit config.py: "target_level": "global" 
python3 main.py

# Test adaptive attack
# Edit config.py: "target_level": "adaptive"
python3 main.py
```

## 📊 Monitoring Attack Effectiveness

Check the console output for:
```
Attack Effectiveness:
- Gradient Direction Change: X.XXX (1.0 = perfect opposite)
- Magnitude Preservation: X.XXX (1.0 = same magnitude)  
- Overall Attack Score: X.XXX (higher = more effective)
- Target Level: [client/cluster/global/adaptive]
```

Higher scores = more effective attacks
Lower scores with clustering = better defense!