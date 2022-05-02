# Texture-synthesis
Implementation of the texture synthesis algorithm described in "Texture Synthesis by Non-parametric Sampling" by A.A. Efros and T.K. Leung.

In addition to this implementation a slight variation of this method is used to de-noise an image with the same approach.

Method description:

![image](https://user-images.githubusercontent.com/35537164/166247656-a5b9a137-3df7-4778-923f-4bfb869fd74c.png)

## Demonstration from the image above
![image](https://github.com/sverrirhd/Texture-synthesis/blob/main/gifs/all_together.gif?raw=true)
The above image demonstrates the tradeoff in quality and speed that is made with kernel size. It also shows how well the method works on different types of data. It performs extremely well on uniformily repeating patterns, slightly worse on more natural, uneven patterns but worst on images without patterns, although it does a decent job of making the last image look close to natural 
