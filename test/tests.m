%% Tests %%             -- Antonio Carretero Sahuquillo uwu --
%                       -- Utilizado para realizarlo:
%                       https://es.mathworks.com/help/deeplearning/ref/alexnet.html
%                       --
%% Carga de datos de las imagenes de entrenamiento y de validación:

%Objeto para "almacenar" las imagenes:
imds = imageDatastore("imgs\","IncludeSubfolders",true,"LabelSource","foldernames");
    %El primer argumento indica la "dirección" donde se encuentran las
    %imágenes.

    %El segundo argumento y tercero van unidos, para indicar que hay
    %subcarpetas.

    %El tercer y cuarto argumento van unidos, para indicar que etiquete a
    %la imagenes con el mismo nombre que tienen las carpetas que las
    %contienen.

    %Salida: Crea un objeto en "imds" (nombre que le he puesto) con las rutas de
    %cada una de las imagenes y sus etiquetas.

%Dividimos estas imagenes en dos objetos de imágenes de entrenamiento y validación:
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
    %La función divide cada imagen del objeto imds. 

    %Con una proporción de 70% para imágenes de entrenamiento (imdsTrain) y el resto de validación.

    %El tercer argumento indica que las divida de manera aleatoria, es
    %decir, que en cada imds habrá tanto imagenes con pulgones como sin
    %pulgones pero estarán correctamente etiquetadas.

%% Carga y ajuste de una red preentrenada --> "alexnet" (vale cualquier otra red para detección de objetos en imágenes).
net = alexnet; %Cargo la arquitectura de la red en la variable net.

    %Esto de aquí está bastante chulo, es para visualizar la red y poder
    %interactuar con ella de forma gráfica, descomentala si quieres verla:
        %analyzeNetwork(net)

    %La primera capa de la red, como se observa descomentando lo de arriba,
    %requiere imágenes de un tamaño de 227 x 227 x 3, donde 3 es el número
    %de canales de color.

%Tamaño de las imágenes de entrada a la red neuronal:
inputSize = net.Layers(1).InputSize; %Cargamos estas dimensiones de entrada en la variable inputsize.
    %Para visualizarlo descomenta: 
        %inputsize
   

%Las últimas tres capas de la red están configuradas para 1000 clases.
%(Porque es un modelo ya entrenado para reconocer 1000 objetos distintos) 
%Puesto que queremos ajustar esta red para que detecte dos clases: Con
%pulgones y Sin pulgones, extraemos estas capas.

%Defino en una nueva variable el número de clases que vamos a tener:
numClasses = numel(categories(imdsTrain.Labels));
    %Para visualizar el número de clases descomentar: 
        %numClasses

%Extracción de las últimas tres capas, guardando todos los layers de la red
%menos las tres útltimas:
layersTransfer = net.Layers(1:end-3);

%Ahora hay que transferir estas capas y añadir otras tres: Capa completamente conectada, capa softmax y por último una capa de clasificación de salida.
%(según la documentación de matlab, yo no entiendo el porqué no soy 100tífico)
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Entrenacionado de la red.
%Antes, tenemos que ajustar las imagenes al tamaño de entrada de la red.
%Para ello utilizo un almacén de datos de imágenes aumentado para cambiar
%automáticamente el tamaño de las imágenes de entrenamiento y hacerle
%algunas modificaciones:
    %voltear aleatoriamente las imágenes de entrenamiento a lo largo del
    %eje vertical.
    %trasladarlas aleatoriamente hasta 30 píxeles horizontal y
    %verticalmente.
%El aumento de datos ayuda a evitar que la red se sobreajuste y memorice
%los detalles exactos de las imágenes de entrenamiento.
pixelRange = [-30 30];

imageAugmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);

%Para las imágenes de validación se realiza automáticamente el ajuste de tamaño sin
%realizar más aumentos de datos ni realizar operaciones adicionales (por razones obvias :3):
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


%Especifico ahora las opciones de entrenamiento:
%[Lo he  copiado de la documentación de matlab]
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%Lo ultimo es para plotear como aprende la red, está muy muy chulo.

%EL ENTRENAMIENTOOOO:
netTransfer = trainNetwork(augimdsTrain,layers,options);
    %Guardo la nueva red entrenada en netTransfer

    %El primer argumento contiene las imagenes de entrenamiento ajustadas y
    %modificadas como mencinado anteriormente.

    %Los layers contienen las capas de la red neuronal, también explicado
    %antes.

    %Y las opciones, pues eso.

%% Clasificación de las imagenes de validación.
[YPred,scores] = classify(netTransfer,augimdsValidation);
%Para clasificar las imágenes con la red entrenada.

%Visualización de las imágenes de visualización clasificadas con la red.
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%Cálculo de la precisión de la clasificación en el conjunto de validación
%(Esto es el número de etiquetas que la red predice correctamente)
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
    %Para visualizar la precisión descomentar:
        %accuracy
