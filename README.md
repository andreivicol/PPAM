Detectarea si conservarea contururilor din imagini folosind filtrul Total Variation

Asemenea filtrului Laplacian, filtrul Total Variation este sensibil la variatiile intensitatilor pixelilor din cadrul unei regiuni a imaginii, putand capta variatiile cauzate de diferentele de valoare intre zonele delimitate de o linie de contur.
Pentru implementarea algoritmului de testare paralela s-a folosit framework-ul CUDA pentru un gpu GTX 960M. Evaluarea performantelor algoritmului paralel in comparatie cu cel serial s-a realizat folosind imagini de rezolutii si dimensiuni diferite, subliniind in acest fel diferentele de timp necesare executiei.

Imaginile de test:
cat.png : 513x595 
goat.png : 1649x1613

Pentru evaluarea imaginilor s-a folosit biblioteca lodepng pentru a incarca imaginea in memorie sub forma unui vector cu valori pe 8 biti fara semn. Pentru a putea accesa in mod natural memoria, vectorul continand imaginea este transformat intr-un array de valori pe 8 biti si incarcat in memoria GPU folosind functiile de acces oferite de CUDA(cudaMalloc pentru crearea unui array cuprinzator si cudaMemcpy pentru transferul valorilor dintr-un buffer in altul).
Masurarea timpilor de executie s-a realizat folosind functionalitatile puse la dispozitie de openCV pentru numararea ciclurilor de procesor si obtinerea perioadei acestuia. Pentru facilitarea si lizibilitatea timpilor obtinuti s-au convertit in milisecunde.

Rezultate obtinute:
- cat.png, avand dimensiunea de 424KB a necesitat 0.0259ms pentru executia algoritmului paralelizat si 1.2765ms pentru executia algoritmului serial;
- goat.png, avand dimensiunea de 1.6MB a necesitat 0.0321ms pentru executia algoritumului paralelizat si 14.865ms pentru executia algoritmului serial;

Bibliografie:
- algoritmul Total Variation: https://www.researchgate.net/publication/262292636_Total_variation_image_edge_detection
- implementarea in C a functiei de obtinere a filtrului: https://github.com/KhosroBahrami/ImageFiltering_CUDA/blob/master/TVFilter/tvFilter.cu
- conversia de la vector 1D la matrice: https://answers.opencv.org/question/81831/convert-stdvectordouble-to-mat-and-show-the-image/
- layout-ul pixelilor produs de lodepng si documentatia aferenta utilizarii bibliotecii: https://github.com/lvandeve/lodepng/blob/master/lodepng.h
