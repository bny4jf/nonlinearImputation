library(stats)
library(msm)
library(np)

#####estimation w.o. optimal weighting matrix
n=2000
nrep=500

imputationseparateresult=matrix(0,nrep,3)
completeresult=matrix(0,nrep,3)
fullresult=matrix(0,nrep,3)
JDresult=matrix(0,nrep,5)

for (l in 1:nrep){
  ####Data generating
  alpha=1
  beta1=1
  beta2=1
  
  u=rtnorm(n,0,1,-3,3)
  v=rtnorm(n,0,1,-3,3)
  z2=rtnorm(n,0,1,-3,3)

  x=1+1*z2+v
  xfull=x

  y=alpha+beta1*x+beta2*z2+u

  missingindicator<-runif(n,0,1)
  m=ifelse(missingindicator>=0.5,0,1)
  x=ifelse(m==1,0,x)
  nonmissingdata<-cbind(x[which(m==0)],z2[which(m==0)],y[which(m==0)]) 
  
  ####imputation, complete, full, and JD estimation
  momentthreeseparate=NA
  momentfourseparate=NA
  
  moment1separate=nonmissingdata[,1]*(nonmissingdata[,3]-1-nonmissingdata[,1]-nonmissingdata[,2])
  bw1separate<-npregbw(formula=moment1separate~nonmissingdata[,2])
  moment2separate=nonmissingdata[,2]*(nonmissingdata[,3]-1-nonmissingdata[,1]-nonmissingdata[,2])
  bw2separate<-npregbw(formula=moment2separate~nonmissingdata[,2])
  
  imputationseparate<-function(par){
    g=c(0,0,0,0)
    for(i in 1:n){
      
      numeratorthree=0
      denominatorthree=0
      for (j in 1:dim(nonmissingdata)[1]){
        numeratorthree=numeratorthree+(nonmissingdata[j,1]*(nonmissingdata[j,3]-par[1]-par[2]*nonmissingdata[j,1]-par[3]*nonmissingdata[j,2]))*(1/sqrt(2*pi)*exp(-0.5*((nonmissingdata[j,2]-z2[i])/(bw1separate$bw))^2))
        denominatorthree=denominatorthree+(1/sqrt(2*pi)*exp(-0.5*((nonmissingdata[j,2]-z2[i])/(bw1separate$bw))^2))
      }
      momentthreeseparate[i]=numeratorthree/denominatorthree
      
      numeratorfour=0
      denominatorfour=0
      for (k in 1:dim(nonmissingdata)[1]){
        numeratorfour=numeratorfour+(nonmissingdata[k,2]*(nonmissingdata[k,3]-par[1]-par[2]*nonmissingdata[k,1]-par[3]*nonmissingdata[k,2]))*(1/sqrt(2*pi)*exp(-0.5*((nonmissingdata[k,2]-z2[i])/(bw2separate$bw))^2))
        denominatorfour=denominatorfour+(1/sqrt(2*pi)*exp(-0.5*((nonmissingdata[k,2]-z2[i])/(bw2separate$bw))^2))
      }
      momentfourseparate[i]=numeratorfour/denominatorfour
      
      g[1]=g[1]+(1-m[i])*x[i]*(y[i]-par[1]-par[2]*x[i]-par[3]*z2[i])
      g[2]=g[2]+(1-m[i])*z2[i]*(y[i]-par[1]-par[2]*x[i]-par[3]*z2[i])
      g[3]=g[3]+(m[i])*momentthreeseparate[i]
      g[4]=g[4]+(m[i])*momentfourseparate[i]
    }
    
    return(t(g)%*%g/(n^2))
    
  }
  
  complete<-function(par){
    g=c(0,0)
    for(i in 1:n){
      g[1]=g[1]+(1-m[i])*x[i]*(y[i]-par[1]-par[2]*x[i]-par[3]*z2[i])
      g[2]=g[2]+(1-m[i])*z2[i]*(y[i]-par[1]-par[2]*x[i]-par[3]*z2[i])
    }
    
    return(t(g)%*%g/(n^2))
    
  }
  
  full<-function(par){
    g=c(0,0)
    for(i in 1:n){
      g[1]=g[1]+xfull[i]*(y[i]-par[1]-par[2]*xfull[i]-par[3]*z2[i])
      g[2]=g[2]+z2[i]*(y[i]-par[1]-par[2]*xfull[i]-par[3]*z2[i])
    }
    
    return(t(g)%*%g/(n^2))
    
  }
  
  JD<-function(par){
    g=c(0,0,0,0)
    for(i in 1:n){
      g[1]=g[1]+(1-m[i])*x[i]*(y[i]-par[1]-par[2]*x[i]-par[3]*z2[i])
      g[2]=g[2]+(1-m[i])*z2[i]*(y[i]-par[1]-par[2]*x[i]-par[3]*z2[i])
      g[3]=g[3]+(1-m[i])*z2[i]*(x[i]-par[4]-par[5]*z2[i])
      g[4]=g[4]+(1-m[i])*(z2[i]*(y[i]-par[1]-par[2]*par[4]-(par[2]*par[5]+par[3])*z2[i]))
    }
    
    return(t(g)%*%g/(n^2))
  }
  
  imputationseparateestimation = optim(par=c(0.8,0.8,0.8),imputationseparate, method='L-BFGS-B', lower= c(0.5,0.5,0.5), upper=c(1.5,1.5,1.5))
  completeestimation = optim(par=c(0.8,0.8,0.8),complete, method='L-BFGS-B', lower= c(0.5,0.5,0.5), upper=c(1.5,1.5,1.5))
  fullestimation = optim(par=c(0.8,0.8,0.8),full, method='L-BFGS-B', lower= c(0.5,0.5,0.5), upper=c(1.5,1.5,1.5))
  JDestimation = optim(par=c(0.8,0.8,0.8,0.8,0.8),JD, method='L-BFGS-B',lower= c(0.5,0.5,0.5,0.5,0.5), upper=c(1.5,1.5,1.5,1.5,1.5))
  
  imputationseparateresult[l,]=imputationseparateestimation$par
  completeresult[l,]=completeestimation$par
  fullresult[l,]=fullestimation$par
  JDresult[l,]=JDestimation$par
  
  print(l)

}

imputationseparatebiasalpha=mean(imputationseparateresult[,1])-1
imputationseparatebiasbeta1=mean(imputationseparateresult[,2])-1
imputationseparatebiasbeta2=mean(imputationseparateresult[,3])-1
imputationseparateMSEalpha=mean((imputationseparateresult[,1]-1)^2)
imputationseparateMSEbeta1=mean((imputationseparateresult[,2]-1)^2)
imputationseparateMSEbeta2=mean((imputationseparateresult[,3]-1)^2)
  
completebiasalpha=mean(completeresult[,1])-1
completebiasbeta1=mean(completeresult[,2])-1
completebiasbeta2=mean(completeresult[,3])-1
completeMSEalpha=mean((completeresult[,1]-1)^2)
completeMSEbeta1=mean((completeresult[,2]-1)^2)
completeMSEbeta2=mean((completeresult[,3]-1)^2)  

fullbiasalpha=mean(fullresult[,1])-1
fullbiasbeta1=mean(fullresult[,2])-1
fullbiasbeta2=mean(fullresult[,3])-1
fullMSEalpha=mean((fullresult[,1]-1)^2)
fullMSEbeta1=mean((fullresult[,2]-1)^2)
fullMSEbeta2=mean((fullresult[,3]-1)^2) 

JDbiasalpha=mean(JDresult[,1])-1
JDbiasbeta1=mean(JDresult[,2])-1
JDbiasbeta2=mean(JDresult[,3])-1
JDMSEalpha=mean((JDresult[,1]-1)^2)
JDMSEbeta1=mean((JDresult[,2]-1)^2)
JDMSEbeta2=mean((JDresult[,3]-1)^2)




