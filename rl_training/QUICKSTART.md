# Feynman-GCPN Quick Start Guide

Welcome to **Feynman-GCPN**! This guide will get you training in 5 minutes.

## 🚀 30-Second Quickstart

```powershell
# 1. Install dependencies
cd rl_training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Start training
python train.py

# 3. Watch it learn (open in browser)
start ../training_viz.html
```

That's it! The agent will start learning to draw Feynman diagrams.

---

## 📋 What to Expect

### First 10 Minutes
- **Episodes 0-50**: Random exploration, negative rewards
- **Console Output**: Episode numbers, rewards, loss values
- **training_viz.html**: Shows current diagram (initially empty/random)

### After 30 Minutes
- **Episodes 50-200**: Agent discovers basic conservation laws
- **Rewards**: Start becoming positive (~5-10)
- **Diagrams**: Simple single-propagator diagrams appear

### After 1-2 Hours
- **Episodes 200-500**: Agent learns valid topologies
- **Rewards**: Converge to ~10-15
- **Diagrams**: Correct e⁻ + e⁺ → μ⁻ + μ⁺ via γ

### Final Result
A valid Feynman diagram saved to `diagrams/current_best.json` that you can import into Feynman Forge!

---

## 🎮 Monitoring

### Option 1: Terminal Output
Watch the console for real-time progress:
```
[Update 10/50] Step: 20480
  Mean Reward: 2.34
  Mean Length: 12.5
  Best Reward: 8.72
  Policy Loss: 0.0234
  Value Loss: 1.2345
```

### Option 2: Training Visualization
Open `training_viz.html` in your browser:
- Auto-refreshes every 2 seconds
- Shows current episode, reward, diagram
- Training history timeline

### Option 3: TensorBoard
```powershell
tensorboard --logdir=logs
```
Then open `http://localhost:6006` for detailed metrics.

---

## 🎨 Try Different Reactions

### Example 1: Muon Decay
```powershell
python train.py --reaction "mu->e+nu_mu+nu_e"
```

### Example 2: Compton Scattering
```powershell
python train.py --reaction "photon+e->photon+e"
```

### Example 3: Quark Scattering
```powershell
python train.py --reaction "u+d->u+d"
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'torch_geometric'"
```powershell
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### "No diagrams appearing in training_viz.html"
- Wait at least 10 episodes (20-30 seconds)
- Check that `diagrams/` folder exists
- Refresh the page manually

### Training is slow
```powershell
# Use GPU (if available)
python train.py --device cuda

# Or reduce model size
python train.py --hidden-dim 64
```

### Rewards are negative
This is normal at the start! The agent is exploring randomly. Give it 50-100 episodes.

---

## 📊 Understanding the Output

### Reward Breakdown
- **+10**: Correct initial/final particles
- **+5**: Valid topology (connected, no dangling lines)
- **-0.5 per vertex**: Complexity penalty
- **-5 per violation**: Conservation law violations

### Target Reward
For `e+e->mu+mu`:
- **Optimal**: ~13-15 (correct diagram with 2-3 vertices)
- **Good**: ~8-12 (mostly correct, minor issues)
- **Poor**: <5 (invalid or incomplete)

---

## 🎓 Next Steps

1. **Let it train** for 100k steps (~1-2 hours)
2. **Check `diagrams/current_best.json`**
3. **Import into Feynman Forge**:
   - Open `feymann.html`
   - Click **Canvas Manager** → **Import**
   - Select the JSON file
4. **View and validate** the diagram
5. **Try more reactions** from `EXAMPLES.md`

---

## 📚 Learn More

- **Full Guide**: Read `README.md` in `rl_training/`
- **Examples**: See `EXAMPLES.md` for advanced usage
- **Methodology**: Check `design4RL.md` for the theory
- **Code**: Explore the Python files (well-commented!)

---

## 💡 Pro Tips

### Faster Training
- Use `--timesteps 500000` for better convergence
- Enable GPU with `--device cuda`
- Increase rollout steps in `training.py`

### Better Diagrams
- Increase physics gate penalty: `--lambda-penalty 10.0`
- Train longer on complex reactions
- Use curriculum learning (train on simple → complex)

### Debugging
- Check TensorBoard for loss trends
- Inspect `diagrams/current_diagram.json` manually
- Enable verbose logging in `training.py`

---

## ✅ Verification Checklist

Before training, verify:
- [ ] Python 3.8+ installed
- [ ] PyTorch installed (`python -c "import torch; print(torch.__version__)"`)
- [ ] PyTorch Geometric installed (`python -c "import torch_geometric"`)
- [ ] CUDA available (optional): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] All dependencies installed: `pip list | grep -E "torch|gymnasium|numpy"`

---

## 🆘 Getting Help

1. **Check the error message** carefully
2. **Read the Troubleshooting section** above
3. **Inspect the code** - it's well-documented!
4. **Check your dependencies** with `pip list`

---

## 🎉 Success Criteria

You'll know it's working when:
✅ Console shows increasing rewards
✅ `training_viz.html` displays diagrams
✅ `diagrams/current_best.json` exists
✅ TensorBoard shows decreasing policy loss
✅ The final diagram is physically valid

---

**Ready? Let's go!**

```powershell
cd rl_training
python train.py
```

Watch the agent learn quantum field theory from scratch! 🚀✨
