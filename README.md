# 🤖 Genderuwo Bot - Intelligent Diamond Collector

<div align="center">
  
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

</div>

<div align="center">
  <h3>🎯 Advanced Multi-Weighted Greedy Algorithm Bot</h3>
  <p>
    An intelligent Etimo Diamonds game bot powered by sophisticated greedy algorithms
    <br />
    <a href="https://github.com/MarioSitepu/Tubes1_Sigma"><strong>📖 Explore Documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MarioSitepu/Tubes1_Sigma/issues">🐛 Report Bug</a>
    ·
    <a href="https://github.com/MarioSitepu/Tubes1_Sigma/issues">💡 Request Feature</a>
  </p>
</div>

---

## 📋 Table of Contents

<details>
  <summary>Click to expand</summary>
  
  1. [🎯 About The Project](#about-the-project)
  2. [🚀 Getting Started](#getting-started)
     - [📋 Prerequisites](#prerequisites)
     - [⚡ Installation & Usage](#installation--usage)
  3. [📊 Project Status](#project-status)
  4. [🔧 Future Improvements](#future-improvements)
  5. [🙏 Acknowledgments](#acknowledgments)
  
</details>

---

## 🎯 About The Project

**Genderuwo Bot** is an advanced AI agent designed to dominate the Etimo Diamonds game using cutting-edge algorithmic strategies. The bot employs a **Multi-Weighted Greedy Algorithm** that combines intelligent decision-making with dynamic heuristics to maximize diamond collection efficiency.

### 🧠 Core Algorithm Features

- **Dynamic Heuristic Greedy (DHG)** approach for optimal move selection
- **Multi-factor weighted decision system** considering:
  - Distance optimization
  - Point value assessment  
  - Strategic resource utilization
- **Advanced game mechanics integration** including red buttons and teleporters
- **Real-time adaptive strategy** based on game state analysis

The heart of our intelligence lies in the `Multi-Weighted` algorithm, meticulously crafted and located in:
```
📁 game/logic/multi_weighted.py
```

---

## 🚀 Getting Started

### 📋 Prerequisites

Ensure you have the following tools installed on your system:

| Tool | Purpose | Download Link |
|------|---------|---------------|
| **Node.js** | JavaScript runtime | [nodejs.org](https://nodejs.org/en) |
| **Python 3.x** | Core language | [python.org](https://python.org) |
| **Docker Desktop** | Containerization | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Yarn** | Package manager | Install via npm |

### ⚡ Installation & Usage

Follow these simple steps to get Genderuwo Bot up and running:

#### 1️⃣ **Download & Setup**
```bash
# Download the source code
git clone https://github.com/MarioSitepu/Tubes1_Sigma.git

# Navigate to project directory
cd Tubes1_Sigma/tubes1-IF2110-bot-starter-pack-1.0.1
```

#### 2️⃣ **Install Dependencies**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Yarn globally (if not already installed)
npm install --global yarn
```

#### 3️⃣ **Launch the Bot**
```bash
# Run Genderuwo Bot with custom configuration
python main.py \
  --logic MultiWeighted \
  --email=multi@email.com \
  --name=Genderuwo \
  --password=123456 \
  --team etimo
```

> 💡 **Pro Tip:** Customize the bot parameters according to your game strategy needs!

---

## 📊 Project Status

<div align="center">
  
![Status](https://img.shields.io/badge/Status-✅%20Complete-brightgreen?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0.1-blue?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-Stable-success?style=for-the-badge)

</div>

The Genderuwo Bot project has reached completion status with all core functionalities implemented and tested. The bot demonstrates consistent performance in diamond collection scenarios with optimized decision-making capabilities.

---

## 🔧 Future Improvements

Our development roadmap includes several exciting enhancements:

### 🎯 **Performance Optimizations**
- Algorithm processing speed improvements
- Code efficiency and memory optimization
- Resource utilization enhancements

### 🧭 **Advanced Navigation**
- Enhanced obstacle avoidance mechanisms
- Basic A* pathfinding algorithm implementation
- Improved movement prediction systems

### 🤖 **Intelligence Upgrades**
- Competitive intelligence analysis
- Enemy behavior prediction models
- Dynamic strategy switching capabilities

### ⚙️ **Technical Refinements**
- Parameter fine-tuning for optimal performance
- Real-time performance monitoring
- Advanced debugging and logging systems

---

## 🙏 Acknowledgments

Special thanks to our talented development team who made this project possible:

<div align="center">

| Developer | GitHub Profile | Contribution |
|-----------|----------------|--------------|
| **Anselmus Herpin Hasugian** | [) | Algorithm Design & Implementation |
| **Mario Fransiskus Sitepu** | [@mario_stp](https://github.com/MarioSitepu) | Core Logic & Optimization |
| **Margareta Angela Manullang** | [) | Testing & Documentation |

</div>

---

<div align="center">
  
**⭐ If you found this project helpful, please consider giving it a star!**

[⬆️ Back to Top](#-genderuwo-bot---intelligent-diamond-collector)

</div>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/MarioSitepu/Tubes1_Sigma.svg?style=for-the-badge
[contributors-url]: https://github.com/MarioSitepu/Tubes1_Sigma/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MarioSitepu/Tubes1_Sigma.svg?style=for-the-badge
[forks-url]: https://github.com/MarioSitepu/Tubes1_Sigma/network/members
[stars-shield]: https://img.shields.io/github/stars/MarioSitepu/Tubes1_Sigma.svg?style=for-the-badge
[stars-url]: https://github.com/MarioSitepu/Tubes1_Sigma/stargazers
[issues-shield]: https://img.shields.io/github/issues/MarioSitepu/Tubes1_Sigma.svg?style=for-the-badge
[issues-url]: https://github.com/MarioSitepu/Tubes1_Sigma/issues
[license-shield]: https://img.shields.io/github/license/MarioSitepu/Tubes1_Sigma.svg?style=for-the-badge
[license-url]: https://github.com/MarioSitepu/Tubes1_Sigma/blob/master/LICENSE
