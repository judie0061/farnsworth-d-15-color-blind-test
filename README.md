# D-15 Digital Color Vision Test

This is a web-based implementation of the **Farnsworth D-15 color vision test**, designed to help identify the **type and severity of color vision deficiency (CVD)**.

The test asks users to **reorder colored caps** to form a continuous hue sequence. Once submitted, it computes clinically relevant metrics:

- **Total Error Score (TES)** – measures deviation from correct order  
- **Confusion Angle** – used to infer type of CVD (e.g., protan, deutan, tritan)  
- **Confusion Index (CI)** – quantifies severity based on comparison to a normal TES

This tool was built as the diagnostic component of a larger project aimed at enhancing accessibility for colorblind users through **personalized image-color correction**.

---

## Quick Start

1. Clone this repo
      ```bash
   git clone https://github.com/judie0061/d15-color-test.git
   cd d15-color-test
  
2. Install dependencies
      ```bash
   pip install flask numpy
3. Open app.py in your preferred IDE and run it.
4. Visit the local link shown in the terminal.
5. Now you can start the test!


## Features
- Interactive drag-and-drop test interface
- Fixed pilot cap and randomized draggable caps
- Real-time computation of:
    - Total Error Score (TES)
    - Confusion Angle (via PCA in CIE u′v′ space)
    - Confusion Index (CI)
 

## Related Project

This test is part of a larger project exploring personalized color correction for colorblind users, where test results directly inform how image colors are adjusted for better visibility.

Link: https://github.com/CZ124/GenAI2025
