---
language:
- en
tags:
- biology
- esm
- protein

---

# Model Card for esmc-600m-2024-12

ESM Cambrian is a parallel model family to our flagship ESM3 generative models. While ESM3 focuses on controllable generation of proteins for therapeutic and many other applications, ESM C focuses on creating representations of the underlying biology of proteins. ESM C scales up data and training compute to deliver dramatic performance improvements over ESM2.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6306e13e5b87d4feaad68405/F6SS--Yt9bLctcEE-Ydcl.png)

The ESM family of models has consistently set the state of the art for protein language models. Now ESM C unlocks a new frontier of performance with scaling up to 6 billion parameters. The ESM C family delivers significant inference time performance gains, matching and even exceeding the capabilities of far larger previous generation models. 

## Usage

Using `ESMC` requires [esm](https://github.com/evolutionaryscale/esm)

```
pip install esm
```

Please refer to the readme and notebooks in the [esm repository](https://github.com/evolutionaryscale/esm) for details on how to use the model.

## License

This repository is under a custom non-commercial [license](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement).
