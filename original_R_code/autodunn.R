#Original R code by Dr. Gul Tokdemir that motivated the current tool and research.
#Apply Kruskal-Wallis test to several attributes in a given data frame

#read data from the required location

data  <-  read.csv("C:/RESEARCH/EXPERIMENT_RUNS/dataset.csv",header=TRUE)
attributeinfo  <-  read.csv("C:/RESEARCH/EXPERIMENT_RUNS/attributeinfo.csv",header=TRUE)

attributeinfo <- data.frame(lapply(attributeinfo, as.character), stringsAsFactors=FALSE)



if(!require(FSA)){install.packages("FSA")}
if(!require(lattice)){install.packages("lattice")}
if(!require(multcompView)){install.packages("multcompView")}
if(!require(rcompanion)){install.packages("rcompanion")}

options(max.print = .Machine$integer.max)

#extract numeric attribute names 

d <- attributeinfo[attributeinfo$attrtype=='numeric'& attributeinfo$factororresponse=='response', ]
numericAttrList <- d[,1]

#extract nominal attribute names 

nominalAttrList <- attributeinfo[attributeinfo$attrtype=='nominal' & attributeinfo$factororresponse=='factor', ]
nominalAttrList <- nominalAttrList$attrname

nominalAttrList


results_Kruskal_Wallis <- data.frame(matrix(nrow=0, ncol=6))

colnames(results_Kruskal_Wallis) <- c("comparisons","factorvar", "chi-squared","parameter","p-value","method")

# define variables to hold results of the test
x1 <- numeric() # statistic
x2 <- integer()   # parameter
x3 <- numeric()   # p.value
x4 <- character()   # method
x5 <- character()   # data.name
x6 <- character()   # response variable-numeric
x7 <- character()   # factor variable-categoric

for(i in nominalAttrList )#for each factor variable 
{ 
  
  for (j in numericAttrList) #for each response variable 
  {  
    
    #   on the  left side of the ~ we always use the dependent variable that we want to compare
    # and on the right side we always use the independent (categorical) variable
    res1<-kruskal.test( eval(parse(text=paste("data$", j, sep = ""))) ~ eval(parse(text=paste("data$", i, sep = ""))) ,  data = data)
    
    x1<-res1$statistic
    x2<-res1$parameter
    x3<-res1$p.value
    x4<-res1$method
    x5<-j
    x6<-i
    
    df1 <-  data.frame(x5,x6,x1,x2,x3,x4,stringsAsFactors=FALSE)
    results_Kruskal_Wallis<-rbind(results_Kruskal_Wallis,df1)
    
    
    #   Dunn test for multiple comparisons of groups if Kruskal-Wallis test is significant
    
    sink('C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt',append=TRUE)
    cat("\n\nMETHOD: Dunn_bh  ",j,i,"\n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    cat("\nComparisons       ", "    Z    ",  "     P.unadj  ","      P.adj  \n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    print(dunnTest( eval(parse(text=paste("data$", j, sep = ""))) ~ eval(parse(text=paste("data$", i, sep = ""))), data=data, kw=TRUE,list=TRUE, method="bh"))
    # str(results_Dunn_bh)
    
    
    cat("\n\nMETHOD: Dunn_hs  ",j,i,"\n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    cat("\nComparisons       ", "    Z    ",  "     P.unadj  ","      P.adj  \n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    print( dunnTest( eval(parse(text=paste("data$", j, sep = ""))) ~ eval(parse(text=paste("data$", i, sep = ""))), data=data, kw=TRUE,list=TRUE, method="hs"))
    cat("\n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    # str( results_Dunn_hs)
    
    cat("\n\nMETHOD: Dunn_bonferroni ",j,i,"\n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    cat("\nComparisons       ", "    Z    ",  "     P.unadj  ","      P.adj  \n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    cat("\nComparisons       ", "    Z    ",  "     P.unadj","      P.adj  \n", file="C:\\RESEARCH\\EXPERIMENT_RUNS\\dunn_test_result.txt", append=TRUE)
    print(dunnTest( eval(parse(text=paste("data$", j, sep = ""))) ~ eval(parse(text=paste("data$", i, sep = ""))), data=data, kw=TRUE, list=TRUE, method="bonferroni"))
    # str(results_Dunn_bonferroni)
    
    sink(NULL)
    
    
  }
}
colnames(results_Kruskal_Wallis) <- c("responsevar","factorvar", "chi-squared","parameter","p-value","method")
rownames(results_Kruskal_Wallis) <- c()
write.table(results_Kruskal_Wallis, file = paste("C:\\RESEARCH\\EXPERIMENT_RUNS\\kruskal_wallis_results.csv"),row.names=FALSE, na="",col.names=TRUE, sep=",")


