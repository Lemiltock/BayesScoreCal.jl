library(ggplot2)
library(ks)

CI_check <- function(act, approx){
    if(is.vector(act)){
        # act is 1d-mat of samples, approx is matrix each col is for an act samp
        N <- length(act)
        res <- rep(0, N)
        for(i in 1:N){
            # Get current sample and approx samples
            curr_samp <- act[i]
            approx_samps <- approx[,i]

            # Density estimate for approx dist
            approx_dist <- kde(approx_samps)

            # Find alpha level of true value
            idx <- order(approx_dist$estimate, decreasing=T)
            dx <- approx_dist$eval.point[2] - approx_dist$eval.point[1]
            finalest <- which.min(abs(curr_samp - approx_dist$eval.point))
            CI <- 0
            for(j in idx){
                # Loop over highest density until reach current value
                CI <- CI + (approx_dist$estimate[j] * dx)
                if(j == finalest){
                    break
                }
            }
            res[i] <- CI
        }
    } else {
        # Treat act as multivariate
        N <- dim(act) # N[1] num samps, N[2] num params
        res <- rep(0, N[1])
        for(i in 1:N[1]){
            # Get current sample and approx samples
            curr_samp <- act[i,]
            approx_samps <- approx[,i,]
            
            # Density estimate for approx dist
            approx_dist <- kde(approx_samps)

            # Find alpha level of true value
            idx <- order(approx_dist$estimate, decreasing=T)
            dx <- 1
            for(k in 1:length(dim(approx_dist$estimate))){
                dx <- dx * (approx_dist$eval.point[[k]][2] - 
                            approx_dist$eval.point[[k]][1])
            }
            finalest <- 0
            for(j in N[2]:1){
                # Get flattened value for matrix
                evalidx <- which.min(abs(curr_samp[j] -
                                         approx_dist$eval.point[[j]]))
                if(j != 1){
                    # Not in the last row so need to mulitply by possible rows
                    nrows <- dim(approx_dist$estimate)[j]
                    finalest <- finalest + ((evalidx - 1) * nrows)
                } else { # j = 1
                    # In final row
                    finalest <- finalest + evalidx
                }
            } 
            CI <- 0
            for(j in idx){
                # Loop over highest density until reach current value
                CI <- CI + (approx_dist$estimate[j] * dx)
                if(j == finalest){
                    break
                }
            }
            res[i] <- CI
        }
    }
    return(res)
}


# Read in true samples
trueB <- as.vector(read.csv("trueB.csv", header=F))$V1
truei <- as.vector(read.csv("truei.csv", header=F))$V1
truejoint <- cbind(trueB, truei)

# Read in uni approx samples
uniB <- read.csv("uniB.csv", header=F)
unii <- read.csv("unii.csv", header=F)

# Read in pre uni approx
preB <- read.csv("preB.csv", header=F)
prei <- read.csv("prei.csv", header=F)

# Read in joint approx samples
joinB <- read.csv("joinB.csv", header=F)
joini <- read.csv("joini.csv", header=F)
joint <- array(rep(0, 1000*1000*2), c(1000, 1000, 2))
for(i in 1:1000){
    for(j in 1:1000){
        joint[i,j,1] <- joinB[i,j]
        joint[i,j,2] <- joini[i,j]
    }
}

# Find pre joint approx samplses
prejoint <- array(rep(0, 1000*1000*2), c(1000, 1000, 2))
for(i in 1:1000){
    for(j in 1:1000){
        prejoint[i,j,1] <- preB[i,j]
        prejoint[i,j,2] <- prei[i,j]
    }
}

# Generate CI
test <- CI_check(truejoint, joint)
testB <- CI_check(trueB, uniB)
testi <- CI_check(truei, unii)
testpre <- CI_check(truejoint, prejoint)
testpreB <- CI_check(trueB, preB)
testprei <- CI_check(truei, prei)

true <- seq(0.05, 0.95, 0.01)
app <- true
appB <- true
appi <- true
apppre <- true
apppreB <- true
appprei <- true
for(i in 1:length(true)){
    app[i] = sum(test < true[i])/1000
    appB[i] = sum(testB < true[i])/1000
    appi[i] = sum(testi < true[i])/1000
    apppre[i] = sum(testpre < true[i])/1000
    apppreB[i] = sum(testpreB < true[i])/1000
    appprei[i] = sum(testprei < true[i])/1000
}

### Quick plot
# plot(c(0,1), c(0,1), type='l')
# lines(app, true, col='blue')
# lines(appB, true, col='green')
# lines(appi, true, col='yellow')
#
# lines(apppre, true, col='red')
# lines(apppreB, true, col='pink')
# lines(appprei, true, col='orange')

 # Generate nice plot
plot_data <- data.frame('Actual coverage'=c(app, appB, appi,
                                            apppre, apppreB, appprei),
                        'Target coverage'=rep(true, 6),
                        'Params'=c(rep('(B, u)', length(app)),
                                   rep('B', length(app)),
                                   rep('u', length(app)),
                                   rep('pre(B, u)', length(app)),
                                   rep('preB', length(app)),
                                   rep('preu', length(app))))

# Setup legend labels
my.labs <- list(bquote(beta ~ "," ~ mu),bquote(beta),bquote(mu),
                bquote("pre" ~ beta ~ "," ~ mu),bquote("pre" ~ beta),
                bquote("pre" ~ mu))

# Get colours for groups
default_palette <- scales::hue_pal()(6)
col_breaks <- c("(B, u)", "B", "u", "pre(B, u)", "preB", "preu")

p1 <- ggplot(plot_data, aes(x=Target.coverage,
                            y=Actual.coverage,
                            colour=Params)) +
        geom_line() +
        geom_abline(slope=1, intercept=0) +
        geom_abline(slope=1, intercept=0.1, linetype="dashed") + 
        geom_abline(slope=1, intercept=-0.1, linetype="dashed") +
        xlim(0, 1) +
        scale_x_continuous(breaks=c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
        ylim(0, 1) +
        scale_y_continuous(breaks=c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
        labs(x="Target coverage", y="Actual coverage", color=NULL) +
        scale_colour_manual(values=default_palette,
                            breaks=col_breaks,
                            labels=my.labs) +
        theme_linedraw()

savename <- "ABM_CI_plot_pre.eps"

ggsave(savename,
       plot=p1,
       device = "eps",
       dpi = 1200,
       width = 20,
       height = 15,
       units = "cm")
