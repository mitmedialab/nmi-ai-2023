import numpy as np
import pandas as pd
import scipy.stats as stats

def hello():
    print("Hello")

def getDataList(data, ivar, dvar, conditions_list):
    ''' Creates a list of arrays, each array corresponding to a different group's values for a particular
    dependent variable. The order of arrays will correspond to the order of conditions for ivar in conditions_list.

    Args:
      data: a pandas dataframe of the data
      ivar: a string, the column name independent variable
      dvar: a string, the column name of the dependent variable
      conditions_list: a list of the group names/identifiers found in ivar 

    Returns:
      datalist: the list of tuples of the data in (group, data) pairs, group 
        being a string and data being a numpy array.
    '''

    datalist = []
    for grp in conditions_list:
        df_grp = data.loc[data[ivar] == grp][dvar].to_numpy()
        datalist.append((grp,df_grp))
    return datalist

def getLongTable(datalist):
    ''' Create a long table of values and their corresponding condition group.

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      table_df: a dataframe of the data consisting of two columns: "Condition" and "Data", 
        where "Condition" is the group/condition name and "Data" is the value. 
    '''

    series = [sr[1].tolist() for sr in datalist] 
    grps = [sr[0] for sr in datalist]

    alldata = []
    allgrps = []

    for grp, data in datalist: 
        allgrps.extend([grp] * len(data))
        alldata.extend(data)

    table_df = pd.DataFrame(list(zip(allgrps, alldata)), 
                       columns = ['Condition', 'Data'])
    
    return table_df

def isMinSampleSize(data, group, column_to_count, MIN_SAMPLE_SIZE = 25, debug=True):
    ''' Converts JSON files of grid node layout and grid line layout to CSV files

    Args:
      JSON: grid nodes
      JSON: grid lines

    Returns:
      Saving grid nodes as CSV
      Saving grid lines as CSV
    '''
  
    flag = True

    # get all counts
    grp = data.groupby(group).count().reset_index()
    grp.rename(columns={ column_to_count:'SAMPLE_SIZE'}, inplace=True)
    #return(grp)

    for index, value in grp['SAMPLE_SIZE'].items():
        if value < MIN_SAMPLE_SIZE: #if any are smaller, it will return false
            flag = False

    if debug:
        print(grp)
        print(f"Meets Min Sample Size of {MIN_SAMPLE_SIZE}?: {flag}")

    return flag 

def isMinSampleSize_grps(data, ivar, MIN_SAMPLE_SIZE = 25):
    ''' Checks if each experimental group has a number of samples above a specified minimum.

    Args:
      data: a pandas dataframe of the data
      ivar: the independent variable to define each sample group
      MIN_SAMPLE_SIZE: the minimum sample size

    Returns:
      flag: True if all samples have more values than the minimum, False otherwise 
    '''
    flag = True

    cts = data[ivar].value_counts()

    for ct in cts:
        if ct < MIN_SAMPLE_SIZE: #if any are smaller, it will return false
            flag = False

    return flag 

def checkSamplesEqual(data, ivar, TOL = 0.15):
    ''' Checks if each experimental group has a roughly equal number of samples.

    Args:
      data: a pandas dataframe of the data
      ivar: the independent variable to define each sample group
      TOL: the percentage tolderance of differences 

    Returns:
      flag: True if all samples are roughly equal, False otherwise 
    '''
    flag = True

    cts = data[ivar].value_counts()

    tolVal = np.mean(cts) * TOL 

    for ct in cts:
        for ct2 in cts: 
            if abs(ct - ct2) > tolVal: #if difference is greater than tolerance, return false
                flag = False

    return flag 

def addProcessedCol(df, col_og, newname, name_dict):
    ''' Adds a new column based on an existing column with a new name and renamed values. Edits the existing dataframe.

    Args:
        df: a pandas dataframe of the data
        col_og: the original column name
        newname: the new column name
        name_dict: a dictionary of names in old:new pairs

    '''
    df[newname] = np.nan 
    for j in df.index:
        val_og = df.loc[j, col_og]
        for item in name_dict:
            if (val_og == item):
                df.loc[j, newname] = name_dict[item]


def plotStats(data, ivar, dvar, conditions_list, phoc, pal, ylims=None, test="tukey", lbl=""):
    '''
    Ref: https://blog.4dcu.be/programming/2021/12/30/Posthoc-Statannotations.html 

    Args: 
        data: pandas dataframe of the data
        ivar: string, column name of the independent variable
        dvar: string, column name of the dependent variable
        conditions_list: list of strings, names of groups for independent variable
        phoc: returned results from tukey or dunn
        pal: dictionary, palettes for conditions
        ylims: tuple of y-axis limits, if any
        test: string, must be tukey or dunn

    '''
    from statannotations.Annotator import Annotator
    import matplotlib.pyplot as plt
    import seaborn as sns

    # format post-hoc tests into a non-reduntant list of comparisons with p value
    if test == "tukey":
        stat_df = pd.DataFrame(data=phoc._results_table.data[1:], columns=phoc._results_table.data[0])
    elif test == "dunn":
        dunn = phoc.set_axis(conditions_list, axis=1)
        dunn.set_axis(conditions_list, axis=0, inplace=True)
        
        remove = np.tril(np.ones(dunn.shape), k=0).astype("bool")
        dunn[remove] = np.nan

        stat_df = dunn.melt(ignore_index=False).reset_index().dropna()
        stat_df.set_axis(["group1", "group2", "p-adj"], axis=1, inplace=True)
    elif test == "none": # no significant results, so no post-hoc test
        # generate dummy list for annotations
        import itertools
        cmbs = itertools.combinations(conditions_list, 2)
        g1 = []
        g2 = []
        pv = []
        for comb in cmbs:
            g1.append(comb[0])
            g2.append(comb[1])
            pv.append(1)
        d = {"group1":g1, "group2":g2, "p-adj":pv}
        stat_df = pd.DataFrame(data=d)
            
    else: 
        print("Error: test must be tukey, dunn, or none")
        return
    
    plt.figure(figsize = (3,4))

    pairs = [(i[1]["group1"], i[1]["group2"]) for i in stat_df.iterrows()]
    p_values = [i[1]["p-adj"] for i in stat_df.iterrows()]

    plt.tight_layout()

    sns.set(style="whitegrid")

    ax = sns.barplot(data=data, x=ivar, y=dvar, order=conditions_list, palette=pal, capsize=.1, alpha=0.6)
    #sns.stripplot(data=data, x=ivar, y=dvar, order=conditions_list, palette=pal, dodge=True, alpha=0.35, ax=ax, jitter=0.15)
    sns.swarmplot(data=data, x=ivar, y=dvar, order=conditions_list, color="black", dodge=True, size=2.5, alpha=0.2, ax=ax)
    ax.grid(False)
    ax.text(x=0.5, y=-0.3, s=lbl, fontsize=20, ha='center', va='bottom', transform=ax.transAxes)
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    annot = Annotator(
        ax, pairs, data=data, x=ivar, y=dvar, order=conditions_list, fontsize=20
    )
    annot.configure(text_format="star", loc="outside")
    annot.set_pvalues_and_annotate(p_values)

def isNormal(datalist, debug = True, type = "Shapiro"):
    ''' Normality Tests. 
    
    ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html

    null hypothesis: sample come from a normal distribution ~ i.e. Normal
    rejecting the null hypothesis means it is NOT normal

    Logic 
    Null hypothesis: Data follows a normal distribution
    alternatie hypothesis: Data do not follow a normal distribution 

    If p<0.05, NOT normal, reject null hypothesis 
    If p>0.05, ARE normal distributions, fail to reject null hypothesis

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      flag: True if all groups are normal, False otherwise 
    '''

    if type == "Shapiro":
        print("Performing Shapiro-Wilk normality test...")
        alpha = 0.05
        print(f"alpha_value: {alpha}")

        flag = True
        for dt in datalist:
            grp = dt[0]
            series = dt[1]
            shapiro_test = stats.shapiro(series)
            print(shapiro_test)
            p = shapiro_test[1]
            w = shapiro_test[0]
            if p < alpha:
                if debug:
                    print(f"Series {grp}: is NOT normal. W:{w}, Pvalue: {p}")
                flag = False
            else:
                if debug:
                    print(f"Series {grp}: IS normal. W:{w}, Pvalue: {p}")
        if debug:
            print(f"Normality Assumption Met? : {flag}")
        return flag

    else:  # D’Agostino and Pearson’s Normality Test
        print("Performing D'Agostino-Pearson's normality test...")
        flag = True
        # alpha = 1e-3 #0.001
        alpha = 0.05
        print(f"alpha_value: {alpha}")

        for dt in datalist:
            grp = dt[0]
            series = dt[1]
            k2, p = stats.normaltest(series)
            if p < alpha:
                if debug:
                    print(f"Series {grp}: is NOT normal. Pvalue: {p}")
                flag = False
            else:
                if debug:
                    print(f"Series {grp}: IS normal. Pvalue: {p}")
        if debug:
            print(f"Normality Assumption Met? : {flag}")
        return flag


def isEqualVariances(datalist, isVeryNotNormal=True):
    """ Check Unequal Sample Sizes
    Check the equality of variances for a variable
    Ref: https://stats.stackexchange.com/questions/135232/bartletts-test-vs-levenes-test
    Ref: https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.
    Bartlett's test is sensitive to departures from normality. That is, if your samples come from non-normal distributions, then Bartlett's test may simply be testing for non-normality.
    The Levene test is an alternative to the Bartlett test that is less sensitive to departures from normality.

    Logic
    Null hypothesis : all samples are from populations with equal variances

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().
      isVeryNotNormal: Boolean, use True if the samples come from non-normal distributions.

    Returns:
        True if all populations have equal variance, False otherwise 
    """

    print("Performing Equal Variances Test...")
    if isVeryNotNormal:
        return LeveneTest(datalist)
    else:
        return BartlettTest(datalist)


def LeveneTest(datalist):
    ''' Helper function for isEqualVariances(). Perform Levene test. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      True if all populations have equal variance, False otherwise 
    '''
    print("Performing Levene Test...")
    args = [sr[1]
            for sr in datalist]  # this returns a list of series

    levene_result = stats.levene(*args)  # [0] statistic, [1] pvalue
    print("homogeneity test:", levene_result)
    if levene_result[1] < 0.05:
        print(f"The populations do NOT have equal variances.")
        return False
    else:
        print(f"The populations have equal variances.")
        return True

def BartlettTest(datalist):
    ''' Helper function for isEqualVariances(). Perform Bartlett test. 

    There are significant deviations from normality, use
    Null hypothesis is that each group has the same variance
    Ref: https://www.statology.org/welchs-anova-in-python/

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      True if all populations have equal variance, False otherwise 
    '''

    print("Performing Bartlett Test of Homogeneity of Variances...")
    args = [sr[1]
            for sr in datalist]  # this returns a list of series
    bartlett_result = stats.bartlett(*args)
    print("Result:", bartlett_result)
    if bartlett_result[1] < 0.05:
        print(f"The populations do NOT have equal variances.")
        return False
    else:
        print(f"The populations have equal variances.")
        return True


def passesBasicAnova(datalist):
    ''' Perform basic one-way ANOVA. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      True if at least one of the means of the groups is significantly different, False otherwise 
    '''
    print("Performing ANOVA")
    args = [sr[1]
            for sr in datalist]  # this returns a list of series
    anova = stats.f_oneway(*args)  # [0] statistic, [1] pvalue
    print(anova)
    if anova[1] < 0.05:
        print(
            "ANOVA found signifance. At least one of the means of the groups is different.")
        return True
    else:
        print("oneway ANOVA: no significance. No significant difference between means of the groups.")
        return False
    # return anova

def anovaP(datalist):
    ''' Perform basic one-way ANOVA and return p value. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      p: the p-value. 
    '''
    args = [sr[1]
            for sr in datalist]  # this returns a list of series
    anova = stats.f_oneway(*args)  # [0] statistic, [1] pvalue
    return anova[1]

def passesAnovaWelch(datalist, var_equal=False):
    ''' Perform ANOVA with Welch Statistic. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      True if at least one of the means of the groups is significantly different, False otherwise 
    '''
    
    import scipy
    from collections import namedtuple
    # https://svn.r-project.org/R/trunk/src/library/stats/R/oneway.test.R
    # translated from R Welch ANOVA (not assuming equal variance)

    args = [sr[1] for sr in datalist]  # this returns a list of series

    F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

    args = [np.asarray(arg, dtype=float) for arg in args]
    k = len(args)
    ni = np.array([len(arg) for arg in args])
    mi = np.array([np.mean(arg) for arg in args])
    vi = np.array([np.var(arg, ddof=1) for arg in args])
    wi = ni/vi

    tmp = sum((1-wi/sum(wi))**2 / (ni-1))
    tmp /= (k**2 - 1)

    dfbn = k - 1
    dfwn = 1 / (3 * tmp)

    m = sum(mi*wi) / sum(wi)
    f = sum(wi * (mi - m)**2) / ((dfbn) * (1 + 2 * (dfbn - 1) * tmp))
    prob = scipy.special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf

    print(F_onewayResult(f, prob))
    if prob < 0.05:
        print("ANOVA with Welch Statistic: SIGNIFICANCE. At least one of the means of the groups is different.")
        return True
    else:
        print("ANOVA with Welch Statistic: NO significance. No significant difference between means of the groups.")
        return False

    # return F_onewayResult(f, prob)

def anovaWelchP(datalist, var_equal=False):
    ''' Perform ANOVA with Welch Statistic and return the p-value.

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      prob: the p-value. 
    '''
    
    import scipy
    from collections import namedtuple
    # https://svn.r-project.org/R/trunk/src/library/stats/R/oneway.test.R
    # translated from R Welch ANOVA (not assuming equal variance)

    args = [sr[1] for sr in datalist]  # this returns a list of series

    F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

    args = [np.asarray(arg, dtype=float) for arg in args]
    k = len(args)
    ni = np.array([len(arg) for arg in args])
    mi = np.array([np.mean(arg) for arg in args])
    vi = np.array([np.var(arg, ddof=1) for arg in args])
    wi = ni/vi

    tmp = sum((1-wi/sum(wi))**2 / (ni-1))
    tmp /= (k**2 - 1)

    dfbn = k - 1
    dfwn = 1 / (3 * tmp)

    m = sum(mi*wi) / sum(wi)
    f = sum(wi * (mi - m)**2) / ((dfbn) * (1 + 2 * (dfbn - 1) * tmp))
    prob = scipy.special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
    return prob

# Krukal Wallis Test by Condition
# Reference (KW): https://data.library.virginia.edu/getting-started-with-the-kruskal-wallis-test/#:~:text=If%20we%20have%20a%20small,different%20distribution%20than%20the%20others.
# Reference(KW): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
# Reference (Dunn): https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn/
# Reference (Dunn) Interpretation: https://www.statology.org/dunns-test-python/

# Logic
# If p>0.05, we cannot reject the null hypothesis. The samples come from the same distirbution, therefore there is no significant difference between the groups.
# If p<0.05, we can reject the null hypothesis. There is a significant difference between the groups.

def Kruskal_Wallis_Test(datalist, alpha=0.05, debug=True):
    ''' Krukal Wallis Test by Condition
    Reference (KW): https://data.library.virginia.edu/getting-started-with-the-kruskal-wallis-test/#:~:text=If%20we%20have%20a%20small,different%20distribution%20than%20the%20others.
    Reference(KW): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    Reference (Dunn): https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn/
    Reference (Dunn) Interpretation: https://www.statology.org/dunns-test-python/

    Logic
    If p>0.05, we cannot reject the null hypothesis. The samples come from the same distirbution, therefore there is no significant difference between the groups.
    If p<0.05, we can reject the null hypothesis. There is a significant difference between the groups. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
        first value: the results of the dunn post-hoc tests if a significant result was found
        second value: string, which post-hoc test was used (dunn in this case)
        third value: string, label of which tests were performed

    '''

    import scikit_posthocs as sp
    import itertools

    phoc = None 

    flag = True
    # alpha = 5e-2 #0.05
    print(f"alpha: {alpha}")

    args = [sr[1] for sr in datalist]  # this returns a list of series
    grps = [sr[0] for sr in datalist]  # list of experimental groups

    # for key in data_dict:
    #   print(key)

    # RUN KW Test
    # Inputs are *individual series, one for each category
    kw_result = stats.kruskal(*args)
    statistic = kw_result[0]
    pvalue = kw_result[1]
    if debug:
        print(kw_result)

    flag = False
    if pvalue < alpha:  # KW returns significant result
        print("Significant Result for Kruskal Wallis.")
    else:
        print("No significant result for Kruskal Wallis.")

    print("Compute the DUNN post-hoc test.")
    dunn = sp.posthoc_dunn(args, p_adjust='bonferroni')
    phoc = dunn

    if debug:
        print(f"{dunn}")
        # print(dunn[1])
        # print(dunn[1][4])
        # print(len(dunn))

    # Print Significance Results for Dunn
    print("Pairwise Dunn P-Values:")
    conds = np.arange(1, len(dunn)+1, 1)
    combs = list(itertools.combinations(conds, 2))

    for i in combs:
        if dunn[i[0]][i[1]] <= 0.001:
            print(
                f"{i}: {grps[i[0]-1]}-{grps[i[1]-1]} : pvalue = {dunn[i[0]][i[1]]} : SIGNIFICANT (***)")
            flag = True
        elif dunn[i[0]][i[1]] <= 0.01:
            print(
                f"{i}: {grps[i[0]-1]}-{grps[i[1]-1]} : pvalue = {dunn[i[0]][i[1]]} : SIGNIFICANT (**)")
            flag = True
        elif dunn[i[0]][i[1]] <= alpha:
            print(
                f"{i}: {grps[i[0]-1]}-{grps[i[1]-1]} : pvalue = {dunn[i[0]][i[1]]} : SIGNIFICANT (*)")
            flag = True
        else:
            # print(f"{i}: {grps[i[0]]}-{grps[i[1]]} : pvalue = {dunn[i[0]][i[1]]} : not significant")
            pass
        

    

    if flag:
        return phoc, "dunn", "Kruskal-Wallis, Dunn posthoc"
    else:
        return None, "none", "Kruskal-Wallis"

def kruskalWallisP(datalist, alpha=0.05, debug=True):
    ''' Krukal Wallis Test by Condition, return p value.
    Reference (KW): https://data.library.virginia.edu/getting-started-with-the-kruskal-wallis-test/#:~:text=If%20we%20have%20a%20small,different%20distribution%20than%20the%20others.
    Reference(KW): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    Reference (Dunn): https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn/
    Reference (Dunn) Interpretation: https://www.statology.org/dunns-test-python/

    Logic
    If p>0.05, we cannot reject the null hypothesis. The samples come from the same distirbution, therefore there is no significant difference between the groups.
    If p<0.05, we can reject the null hypothesis. There is a significant difference between the groups. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
        pvalue: the p-value.

    '''

    import scikit_posthocs as sp
    import itertools

    phoc = None 

    flag = True
    # alpha = 5e-2 #0.05

    args = [sr[1] for sr in datalist]  # this returns a list of series
    grps = [sr[0] for sr in datalist]  # list of experimental groups

    # for key in data_dict:
    #   print(key)

    # RUN KW Test
    # Inputs are *individual series, one for each category
    kw_result = stats.kruskal(*args)
    statistic = kw_result[0]
    pvalue = kw_result[1]
    return pvalue

def TukeyTest(datalist):
    ''' Perform Tukey Test to find which groups have a significant difference. Prints results. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
      tukey: tukey results

    '''
    print("Performing Tukey Post-Hoc Test...")
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    long_table = getLongTable(datalist)
    
    tukey = pairwise_tukeyhsd(endog=long_table['Data'].to_numpy(), groups=long_table['Condition'], alpha=0.05)
    print(tukey)
    return tukey
    

def StatTest_checkSamples(datalist):
    ''' Helper function for StatTest. 

    Args:
      datalist: the list of tuples of the data in (group, data) pairs. See getDataList().

    Returns:
        first value: the results of the post-hoc tests if a significant result was found
        second value: string, which post-hoc test was used
        third value: string, label of which tests were performed

    '''
    if isEqualVariances(datalist):
        print('[StatTest] Assumption of Equal Variances was met. Run basic ANOVA.')
        print ('---------------------------------------------')
        if passesBasicAnova(datalist):
            print('[StatTest] Basic ANOVA found significance. Run Tukey post-hoc test.')
            print ('---------------------------------------------')
            phoc = TukeyTest(datalist)
            return phoc, "tukey", "basic ANOVA, Tukey posthoc"
        else:
            print("[StatTest] Tukey found NO statistically significance found between groups.")
            print ('---------------------------------------------')
            phoc = TukeyTest(datalist)
            return None, "none", "basic ANOVA"
    else: 
        print('[StatTest] Assumption of Equal Variances was violated. Run ANOVA with Welch Statistic.')
        if passesAnovaWelch(datalist):
            print('[StatTest] ANOVA with Welch Statistic found significance. Run Tukey post-hoc test.')
            print ('---------------------------------------------')
            phoc = TukeyTest(datalist)
            return phoc, "tukey", "ANOVA with Welch, Tukey posthoc"
        else:
            print("[StatTest] Tukey found NO statistically significance found between groups.")
            print ('---------------------------------------------')
            phoc = TukeyTest(datalist)
            return None, "none", "ANOVA with Welch"


def StatTest(data, ivar, dvar, conditions_list, viz=False, pal=None, ylims=None):
    ''' Main function for running statistical tests. Only works for one question/column of data at a time. 
    Will print results. 

    Args:
      data: a pandas dataframe of the data
      ivar: a string, the column name independent variable
      dvar: a string, the column name of the dependent variable
      conditions_list: a list of the group names/identifiers found in ivar
      viz: boolean. True if plotting the results is desired
      pal: dictionary, palettes for conditions

    Returns: 
      p: the p-value
      tst: a string indicating which statistical test was used

    '''
    datalist = getDataList(data, ivar, dvar, conditions_list)

    results = None
    
    if isMinSampleSize_grps(data, ivar):
        # check sample sizes roughly equal
        print('[StatTest] Minimal Sample Size was met. Check if sample sizes are roughly equal.') 
        print ('---------------------------------------------')
        results = StatTest_checkSamples(datalist)

    else: 
        print ('[StatTest] Minimal Sample Size was not met. Proceed to check Normality Assumption.')
        print ('---------------------------------------------')
        if isNormal(datalist):
            print('[StatTest] Normality Assumption was met. Check if sample sizes are roughly equal.')
            print ('---------------------------------------------')
            results = StatTest_checkSamples(datalist)
        else:
            print ('[StatTest]  Normality Assumption was not met. Proceed with Kruskal-Wallis Test')
            print ('---------------------------------------------')
            results = Kruskal_Wallis_Test(datalist)

    # plot results
    if viz:
        plotStats(data, ivar, dvar, conditions_list, results[0], pal, ylims=ylims, test=results[1], lbl=results[2])

    # get p-value
    tst = results[2]
    p = 1
    if "ANOVA with Welch" in tst:
        p = anovaWelchP(datalist)
    elif "basic ANOVA" in tst:
        p = anovaP(datalist)
    elif "Kruskal-Wallis" in tst:
        p = kruskalWallisP(datalist)

    return p, tst
