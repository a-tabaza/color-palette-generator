# Color Palette Generator (Work in Progress)

Trying out K-Means Clustering to generate color palettes from images. Logging my experiments. I'd like to deploy this as a web app once I'm happy with the results.

## Experimental Results

### K-Means Clustering (Ensemble of K-Means and MiniBatchKMeans)

Palette Sorted by Luminance

- *k* = 8
![Chungking Express](chungking_express.png)
Still from *Chungking Express* (1994) by Wong Kar-Wai
- *k* = 8
![Fallen Angels](fallen_angels.jpg)
Still from *Fallen Angels* (1995) by Wong Kar-Wai
- k = 12
![Drive](drive.jpg)
Still from *Drive (2011)* by Nicolas Winding Refn

### Randomly Sampled Colors from Larger Number of Clusters

Stills from *Parasite (2019)* by Bong Joon-ho

- *k* = 36

![Parasite](parasite_36.png)

- *k* = 36, 8 random colors are selected for palette
![Parasite](randomly_sampled_parasite.png)

- *k* = 36, full palette
![Parasite](36_kmeans.png)

- *k* = 1, but image was segmented into 36 regions to generate palette
![Parastite](36_segmented.png)

### Effect of Scale on Palette

K-Means Clustering (Ensemble of K-Means and MiniBatchKMeans)

*k* = 8

- Scaling Factor = 0.5
![Parasrite](parasite_05.png)
- Scaling Factor = 0.6
![Parasite](parasite_06.png)
- Scaling Factor = 0.7
![Parasite](parasite_07.png)
- Scaling Factor = 0.8
![Parasite](parasite_08.png)
- Scaling Factor = 0.9
![Parasite](parasite_09.png)
