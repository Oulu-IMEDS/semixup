library(lme4)

filename <- './r/data_1000_compare_semixup_cvgg2hv_to_sl_cvgg2hv.csv'

df <- read.csv2(filename, stringsAsFactors=FALSE)
df$x <- as.numeric(df$x)
m1 <- glmer(method ~ x + (1|center/side), data=df, family=binomial)
summary(m1)

m2 <- glmer(method ~ x + (1|center), data=df, family=binomial)
summary(m2)

m3 <- glmer(method ~ x + (1|side), data=df, family=binomial)
summary(m3)

m4 <- glmer(method ~ x + (1|center) + (1|side), data=df, family=binomial)
summary(m4)

m5 <- glmer(method ~ x + (1|id), data=df, family=binomial)
summary(m5)

m6 <- glmer(method ~ x + (1|center/side) + (1|id), data=df, family=binomial)
summary(m6)