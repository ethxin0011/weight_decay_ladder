# WDLadder report (through stage 5)

**Outcome:** O3 non-replication: Muon-SW shows no gain over const decay at MAI hparams on the MoE ladder

## Gate 1 (untuned, MAI hparams, L3 TPP120 cosine)
```
{
 "decision": "FAIL",
 "gain_mean": -0.7282254403899429,
 "gains": [
  -1.4914263985852376,
  0.03497551780535191
 ],
 "seed_noise": 1.527420208388772,
 "threshold": 3.054840416777544,
 "gain_tpp30": 0.046268982280819326,
 "const": [
  1.3225998426909114,
  1.3215815506927289
 ],
 "scaled": [
  2.814026241276149,
  1.286606032887377
 ],
 "speedup_untuned": 0.012136966150795871,
 "note": "PASS -> stage 2; WEAK -> stage 2 grid only (skip GPU until gate 2); FAIL -> stop, write the negative replication note (Muon-SW gain absent on MAI-style MoE ladder)"
}
```
## edge_extension
```
[
 {
  "method": "const",
  "lr_mult": 0.5,
  "prod_mult": 8.0
 },
 {
  "method": "const",
  "lr_mult": 1.0,
  "prod_mult": 8.0
 },
 {
  "method": "const",
  "lr_mult": 2.0,
  "prod_mult": 8.0
 },
 {
  "method": "scaled",
  "lr_mult": 0.5,
  "prod_mult": 16.0
 },
 {
  "method": "scaled",
  "lr_mult": 1.0,
  "prod_mult": 16.0
 },
 {
  "method": "scaled",
  "lr_mult": 2.0,
  "prod_mult": 16.0
 }
]
```
## speedup_at_fitted_opt
```
{}
```
## wd_ratio_wsd
```
0.28702679510801843
```
## K_rel_wsd
```
{
 "const": 0.09997661506325203,
 "scaled": 0.027443297687062038
}
```
## hypothesis_verdict
```
{
 "wsd": {
  "measured": 0.28702679510801843,
  "ci": [
   0.19135119673867895,
   0.43054019266202764
  ],
  "consistent": [],
  "closest": "H_none"
 }
}
```
## scale_trend
```
{
 "const": {
  "L3": NaN,
  "L4": 0.2532447175918854,
  "L6_pred": 0.2532447175918854
 },
 "scaled": {
  "L3": NaN,
  "L4": 0.41400211816737426,
  "L6_pred": 0.41400211816737426
 }
}
```
## L6
```
{
 "const": {
  "best": 1.0210189581611806,
  "at_pred": 1.0223035193378247,
  "regret": 0.0012845611766441056,
  "fitted": 0.3311737509155501,
  "pred": 0.2532447175918854
 },
 "scaled": {
  "best": 1.0159213188915845,
  "at_pred": 1.018073842502537,
  "regret": 0.002188915474541586,
  "fitted": 0.6316725094739677,
  "pred": 0.41400211816737426
 },
 "tuned_gap": 0.005097639269596055
}
```
## hypotheses
```
{
 "cos": {
  "H_none": 1.0,
  "H_int": 1.363998845699143,
  "H_end": 10.0,
  "r": 0.7331384503389601
 },
 "wsd": {
  "H_none": 1.0,
  "H_int": 1.045645743254893,
  "H_end": 10.0,
  "r": 0.9563468377800625
 }
}
```
## seed_sd
```
0.6304382991170936
```
## fits.json (MAI multipliers: lr = mult x LR*_MAI, LR*WD = mult x 17.04/steps)
```
{
 "L3_cos_2d": {
  "const": {
   "coef": [
    1.5689143791206681,
    -0.009854376854027319,
    0.03775629756309432,
    -0.3470853071520005,
    -0.0072089163909109626,
    -0.03385020463405501
   ],
   "rmse": 0.21438632901616295,
   "n": 18,
   "convex": false,
   "boot_ok": 0.0025
  },
  "scaled": {
   "coef": [
    1.4997745814864971,
    -0.005113923966425474,
    0.025285924224342508,
    -0.29152841192752377,
    -0.0051722202249183365,
    -0.024026051704037293
   ],
   "rmse": 0.19698659073663566,
   "n": 21,
   "convex": false,
   "boot_ok": 0.0035
  }
 },
 "L3_wsd_1d": {
  "const": {
   "n": 5,
   "convex": true,
   "coef": [
    1.3551111330284888,
    0.06608056704886384,
    0.015032071238133493
   ],
   "prod_mult": 0.11102653051250962,
   "bpb": 1.282489049346134,
   "prod_ci": [
    0.10302597704165986,
    0.11945581068325778
   ],
   "edge": true
  },
  "scaled": {
   "n": 5,
   "convex": true,
   "coef": [
    1.3385164398282563,
    0.051669842640359204,
    0.0074967145630047555
   ],
   "prod_mult": 0.031867589224968254,
   "bpb": 1.249485017595161,
   "prod_ci": [
    0.01887732543677093,
    0.04789157371256853
   ],
   "edge": true
  }
 },
 "L4_cos_1d": {
  "const": {
   "n": 5,
   "convex": true,
   "coef": [
    1.2241328132149902,
    0.051165279235407236,
    0.018627245046303878
   ],
   "prod_mult": 0.2532447175918854,
   "bpb": 1.188997641686275,
   "prod_ci": [
    0.24820118422982806,
    0.2586105215376482
   ],
   "edge": false
  },
  "scaled": {
   "n": 5,
   "convex": true,
   "coef": [
    1.1939319229548482,
    0.0207372895333941,
    0.011757376873406097
   ],
   "prod_mult": 0.41400211816737426,
   "bpb": 1.1847879790755695,
   "prod_ci": [
    0.361316953943537,
    0.46844922737932115
   ],
   "edge": true
  }
 },
 "pred_L6": {
  "const": {
   "prod_mult": 0.2532447175918854,
   "from_L3": NaN,
   "from_L4": 0.2532447175918854
  },
  "scaled": {
   "prod_mult": 0.41400211816737426,
   "from_L3": NaN,
   "from_L4": 0.41400211816737426
  }
 },
 "pred_L9": {
  "const": {
   "prod_mult": 0.2532447175918854,
   "from_L3": NaN,
   "from_L4": 0.2532447175918854
  },
  "scaled": {
   "prod_mult": 0.41400211816737426,
   "from_L3": NaN,
   "from_L4": 0.41400211816737426
  }
 },
 "L6_1d": {
  "const": {
   "n": 4,
   "convex": true,
   "coef": [
    1.0425456764806758,
    0.03892795905129526,
    0.017612674114614164
   ],
   "prod_mult": 0.3311737509155501,
   "bpb": 1.0210357969184076,
   "prod_ci": [
    0.3215344287810527,
    0.3405175297631301
   ],
   "edge": true
  },
  "scaled": {
   "n": 4,
   "convex": true,
   "coef": [
    1.0184727090858252,
    0.011266308488016442,
    0.012262403103373657
   ],
   "prod_mult": 0.6316725094739677,
   "bpb": 1.0158849270279955,
   "prod_ci": [
    0.5972734016553728,
    0.6658691948576217
   ],
   "edge": true
  }
 }
}
```