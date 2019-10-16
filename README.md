Machine learning custom made model with custom dataset, with lambda code to form a landmarks detection micro-service.
Detection microservice to detect 68 landmarks point using dlib library, code runs in python but the model can also be used in c++. Real time detection of landmarks with milliseconds precision if the library is used with haar cascade opencv face detection rather than using dlib mmod detector.

Landmark model is created with custom dataset annotating the images by hand and with neural networks(FAN). This model is different from the dlib's default model but offers same accuracy(~97%).
