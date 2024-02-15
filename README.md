# ChessGPT2
I instruction fine-tuned Llama 2 13B-hf Large Language Model (LLM) to play chess. I used BitsAndBytes to load the llama 2 in 4-bit int quantization and LoRA to fine-tune low-rank adaptors. I also utilized Hugging Face's accelerate package to be able to train on multiple GPUs.

I added a simple instruction `Predict the next move in this chess game`` as a prompt to the beginning of each game sample.

## Dastaset

I downloaded November archive of lichess.org open database. It had moves and other information of 92,389,636 online played chess games. For start, I randomly selected 100,000 games for training and 10,000 for validation.

I removed annotations on games records so that each train/val example would have only moves and the winner.

```
Predict the next move in this chess game: 1. e4 c6 2. Bc4 d5 3. exd5 cxd5 4. Bb5+ Nc6 5. Bxc6+ bxc6
6. Nc3 Nf6 7. Nf3 e5 8. Nxe5 Qd6 9. d4 Nd7 10. Nxd7 Bxd7 
11. Be3 Qb4 12. Qb1 Rb8 13. Bd2 Qxb2 14. Qxb2 Rxb2 15. 
O-O Rxc2 16. Rfd1 Bb4 17. Rab1 Bxc3 18. Bxc3 Rxc3 19. Rb8
+ Ke7 20. Rxh8 Be8 21. Rxh7 g6 22. Rh3 Rc2 23. a4 Ra2 
24. Re3+ Kd7 25. Rde1 Rxa4 26. Rxe8 Kc7 27. R8e7+ Kb6 
28. Rxf7 Rxd4 29. Rg7 a5 30. Rxg6 a4 31. h3 a3 32. Ra1 
Ra4 33. Rg3 d4 34. Rgxa3 Rb4 35. Rd3 c5 36. Re1 Ka5 37. 
Ra3+ Kb6 38. Rea1 Kc6 39. Ra6+ Kd5 40. R6a5 d3 41. Rd1 
Kd4 42. Ra3 c4 43. h4 c3 44. h5 c2 45. Rc1 Rb1 46. Ra4+ 
Kc3 47. Ra3+ Kd2 48. Rf1 Rxf1+ 49. Kh2 c1=Q 50. Ra2+ Kc3
51. h6 Qxh6+ 52. Kg3 Qg6+ 53. Kf3 Qf6+ 54. Kg3 Qe5+ 55. 
Kf3 Qe2+ 56. Kg3 d2 57. Ra3+ Kb2 58. Re3 Qxf2+ 59. Kh3 
Qxe3+ 60. Kh2 Qg1+ 61. Kg3 Qe3+ 62. Kh2 d1=Q 63. g3 Qh5
+ 64. Kg2 Qf2# 0-1
```

## Training

<p align="center">
  <img src="https://github.com/HosseinEbrahimiK/ChessGPT2/blob/main/logs/eval_loss.png"/>
</p>