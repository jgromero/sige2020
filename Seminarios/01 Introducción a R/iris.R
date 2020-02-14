require(ggplot2)

# Cargar datos
iris_df <- read.table(file = "iris.csv", header = TRUE)
head(iris)

# Gráficos básicos
hist(iris_df$sepal.length)
plot(petal.length ~ petal.width, data = iris_df)

# Gráficos básicos con ggplot2
ggplot(data = iris_df) + geom_histogram(aes(x = sepal.length), bins = 10)
ggplot(data = iris_df) + geom_density(aes(x = sepal.length))
ggplot(data = iris_df) + geom_point(aes(x = petal.length, y = petal.width))

# ggplot2: crear objetos gráficos y añadir capas
g <- ggplot(data = iris_df)
g + geom_point(aes(x = petal.length, y = petal.width, color = factor(class)))
g + geom_point(aes(x = petal.length, y = petal.width, color = factor(class))) + facet_wrap(~factor(class))
g + geom_point(aes(x = petal.length, y = petal.width, color = factor(class), shape = factor(class)))

# formatear gráfico
g + geom_point(aes(x = petal.length, y = petal.width, color=factor(class))) + 
  labs(x = "Petal Length", y = "Petal Width") +  
  scale_color_discrete(name ="Clase", labels=c("Iris Setosa", "Iris Versicolor", "Iris Virginica"))
