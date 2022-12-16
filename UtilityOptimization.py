import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider, Checkbox


class Functions():
    """Класс типовых фукнций полезности. Все входящие данные (кроме параметров) - массивы numpy"""
    def Power(vector: np.array, alpha=1, maximize = True):
        """
        Степенная функция полезности. 
        Значение alpha != 1 указывает на неличениность функции. 
        При alpha > 1 функция выпуклая, при alpha в интервале (0;1) вогнутая."""
        vector = np.array(vector)
        max1 = vector.max()
        min1 = vector.min()
        diff = max1 - min1
        if(maximize):
            return ((vector - min1)/diff)**alpha
        else:
            return ((max1 - vector)/diff)**alpha

    def Logistic(vector : np.array, beta=1, c=0, maximize=True):
        """
        Beta -- вычисляется по формуле (степень нелинейности функции)
        с -- точка перегиба. В этой точке значение функции -- 0.5
        """
        vector = np.array(vector)
        if(maximize):
            beta = -beta
        return 1/(1 + np.exp(beta*(vector - c)))
    
    def LogisticCMD(vector : np.array, c=0, m=1, d=1, maximize=True):
        """
        Beta -- вычисляется по формуле (степень нелинейности функции)
        с -- точка перегиба. В этой точке значение функции -- 0.5
        """
        vector = np.array(vector)
        beta = Functions.Beta(d,m)
        if(maximize):
            beta = -beta
        return 1/(1 + np.exp(beta*(vector - c)))

    def Beta(d, m):
        """
        Параметр d в интервале (0; 0.5) задает максимальное отклонение от предельных значений 0 и 1 
        на концах интервала [c – m; c + m], симметричное относительно точки c."""

        return np.log(1/d -1)/m


    def Bell(vector, beta=1, c=0):
        """Колоколообразная функция полезности. c - координата вершины, beta - ширина колокола"""
        return np.exp(-(vector-c)**2*beta)
    

class Show:
    """
    Класс визуализации. Сигнатуры функций совпадают с сигнатурами функций из Functions. Пробрасывает параметры и строит графики.
    """
    def UtilityFunction(x, y):
        """
        Функция для построения ФП по точкам. 
        Формат входных данных x [1,2,3], y=[0.2, 0.5, 0.8]
        """
        min = x.min()
        diffx = (x.max() - x.min())/100
        diffy=(y.max() - y.min())/100
        punctx = np.arange(x.min(), x.max(), diffx)
        puncty = np.arange(y.min(), y.max(), diffy)
        sns.lineplot(x=punctx, y=puncty, linestyle='--')
        sns.lineplot(x=x, y=y, marker='o')
        plt.grid()
        
    def Power(y, y_points, alpha = 1, maximize=True):
        """
        Степенная функция полезности. 
        y -- условно непрерывный аргумент (с малым приращением).
        y_points -- дискретные точки из y, которые особо интересны. Будут выделены на графике.
        """
        Y = Functions.Power(y, alpha, maximize)        
        sns.lineplot(x=y, y=Y)
        punct = Functions.Power(y, 1, maximize)
        sns.lineplot(x=y, y=punct, linestyle='--')
        plt.grid()
        ret = Functions.Power(y_points, alpha, maximize)
        plt.scatter(y_points, ret, color='blue', s=40, marker='o')
        return ret

    def Logistic(y, y_points, beta=1, c=0, maximize=True):
        """
        Логистичекая функция полезности. 
        y -- условно непрерывный аргумент (с малым приращением).
        y_points -- дискретные точки из y, которые особо интересны. Будут выделены на графике.
        """
        Y = Functions.Logistic(y, beta, c, maximize)
        sns.lineplot(x=y, y=Y)
        punct = Functions.Power(y, 1, maximize)
        sns.lineplot(x=y, y=punct, linestyle='--')
       
        ret = Functions.Logistic(y_points, beta, c, maximize)
        plt.scatter(y_points, ret, color='blue', s=40, marker='o')
        plt.grid()
        return ret
        
    def LogisticCMD(y : np.array, c=0, m=1, d=1, maximize=True):
        """
        Логистическая функция в другой форме параметризации. 
        m - крутизна вдоль абсцисс
        d - крутизна вдоль ординат
        """
        Y = Functions.LogisticCMD(y, c, m, d, maximize)
        sns.lineplot(x=y, y=Y)
        punct = Functions.Power(y, 1, maximize)
        sns.lineplot(x=y, y=punct, linestyle='--')
        plt.grid()
        
    def Bell(y, y_points, beta=1, c=0):
        """
        Колоколообразная функция
        y -- условно непрерывный аргумент (с малым приращением).
        y_points -- дискретные точки из y, которые особо интересны. Будут выделены на графике.
        """
        Y = Functions.Bell(y, beta, c)        
        sns.lineplot(x=y, y=Y, color='blue')

        ret = Functions.Bell(y_points, beta, c)
        plt.scatter(y_points, ret, color='blue', s=40, marker='o')
        plt.grid()
        return ret
        
    def ContributionDiagram(norm, contribution):
        """Отображает диаграмму нормы вклада и вклада. Первый параметр -- норма вклада в процентах, второй -- вклад в процентах"""
        norm = np.round(norm, 2);
        contribution = np.round(contribution, 2);
        
        fig, ax = plt.subplots()
        offset=0.4

        norm_label ='Норма вклада ' + str(norm) + '%'
        value_label = 'Вклад ' + str(contribution) + '%'

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        data1 = np.array([100-contribution, contribution])
        data2 = np.array([100-norm, norm])
        cmap = plt.get_cmap("tab20b")
        b_colors = cmap(np.array([7, 0]))
        sm_colors = cmap(np.array([7, 2]))

        wed2, a = ax.pie(data1, radius=1, colors=b_colors, wedgeprops=dict(width=offset, edgecolor='w'), labels=['', value_label])
        wed, b = ax.pie(data2, radius=1-offset, colors=sm_colors, wedgeprops=dict(width=offset, edgecolor='w'), labels=[norm_label,''])
        
        
        
class UI:
    """
    Класс взаимодействия с пользователем. 
    """
    
    def __init__(self):
        pass;

    def ReadCsv(self, path:str, delimiter=';'):
        """Прочитать файл в формате CSV. path - путь к файлу, delimiter - разделитель (по-умолчению ;)"""
        self.df = pd.read_csv(path, delimiter=delimiter)
        return self.df

    def SelectFunctions(self):
        """Выбор функций для сопоставления"""
        df = self.df
        self.wids = {}
        maxl = len(max(df.columns, key=len)) + 5
        for i in df.columns:
            form_items = [Label(value=i.ljust(maxl, ' '), layout=Layout (width = '7%')),
                          Dropdown(description='Функция', options=[('Логистическая', Functions.Logistic), ('Степенная', Functions.Power), ('Колокообразная', Functions.Bell)],
                                  layout=Layout (width = '20%')
                                  ), 
                          Checkbox(value =False,description='maximize',indent=False, layout=Layout (width = '6%', height ='5px')),
                          FloatText(description='alpha/beta=', layout=Layout (width = '15%', height ='5px'), value=0.5), 
                          FloatText(description='c (logistic)=', layout=Layout (width = '15%', height ='5px'), value=df[i].mean()),
                          FloatText(description='Вес=', layout=Layout (width = '15%', height ='5px'), value=1/len(df.columns))
                         ]


            w = widgets.HBox(form_items)
            self.wids[i] = form_items
            display(w)

    def __Rank(array):
        """Вывод ранга (полностью линейный)"""
        array = np.array(array)
        temp = array.argsort()
        temp = np.flip(temp, 0)
        ranks = temp.argsort() 
        ranks += 1
        return ranks
        #print(ranks)


    def ShowEstimates(self):
        """Просмотр оценок и рангов по функциям полезности"""
        df = self.df
        wids = self.wids

        dfc = df.copy()
        for i in df.columns:
            dfc[i] = UI.__ApplyFunction(df[i], wids[i])

        weights = UI.__GetWeights(wids)
        self.weights = weights
        self.df_only_est = dfc.copy()
        dfc['Оценки'] = pd.Series(UI.__Additive(weights, dfc))
        dfc['Ранги'] = pd.Series(UI.__Rank(dfc['Оценки']))
        #self.estimated_df = dfc
        self.dfc = dfc
        print(dfc)

    def SelectFutureToChange(self):
        """Выбор признака для редактирования параметров его функции полезности"""
        self.change_future=Dropdown(description='Признак', options=self.df.columns, layout=Layout (width = '20%'))
        display(self.change_future)
        
    def __RecalcBorders(self):
        col = self.change_future.value
        self.col = col
        self.maxi = self.wids[self.change_future.value][2].value
        self.y = np.array(self.df[col])
        self.mi = self.y.min() - self.y.mean()
        self.ma = self.y.max() + self.y.mean()
        self.step = (self.ma - self.mi)/500
        
        self.power_step = (self.y.max() - self.y.min())/500
        
    def ChangeParameters(self):
        """
        Функция вызывает интерфейс изменения параметров фукнции полезности
        """
        self.__RecalcBorders()
        print('Функция полезности признака ' + self.change_future.value)
        func = self.wids[self.change_future.value][1].value
        if(func == Functions.Power):
            interact(self.__FPow, alpha=(0,3,0.01))
        if(func == Functions.Logistic):
            interact(self.__FLog, beta=(0, 3, 0.01), c=(-self.mi , self.ma, 0.1))
        if(func == Functions.Bell):
            interact(self.__FBell, beta=(0, 3, 0.01), c=(-self.y.min() , self.y.max(), 0.1))
            
    def __FLog(self, beta, c):
        y = np.arange(self.mi, self.ma, self.step)
        
        newEstimates =Show.Logistic(y=y, y_points=self.y, beta=beta, c=c, maximize=self.maxi)
        
        self.dfc[self.col] = pd.Series(newEstimates)
        self.df_only_est[self.col] = pd.Series(newEstimates)
        self.dfc['Оценки'] = pd.Series(UI.__Additive(self.weights, self.df_only_est))
        self.dfc['Старые ранги'] = self.dfc['Ранги']
        self.dfc['Ранги'] = pd.Series(UI.__Rank(self.dfc['Оценки']))
        print(self.dfc)
        
    def __FPow(self, alpha):    
        y = np.arange(self.y.min(), self.y.max(), self.power_step)
        newEstimates = Show.Power(y=y, y_points=self.y, alpha=alpha, maximize=self.maxi)
        
        self.dfc[self.col] = pd.Series(newEstimates)
        self.df_only_est[self.col] = pd.Series(newEstimates)
        self.dfc['Оценки'] = pd.Series(UI.__Additive(self.weights, self.df_only_est))
        self.dfc['Старые ранги'] = self.dfc['Ранги']
        self.dfc['Ранги'] = pd.Series(UI.__Rank(self.dfc['Оценки']))
        print(self.dfc)
    
    def __FBell(self, beta, c):    
        y = np.arange(self.y.min(), self.y.max(), self.power_step)
        newEstimates = Show.Bell(y=y, y_points=self.y, beta=beta, c=c)
        
        self.dfc[self.col] = pd.Series(newEstimates)
        self.df_only_est[self.col] = pd.Series(newEstimates)
        self.dfc['Оценки'] = pd.Series(UI.__Additive(self.weights, self.df_only_est))
        self.dfc['Старые ранги'] = self.dfc['Ранги']
        self.dfc['Ранги'] = pd.Series(UI.__Rank(self.dfc['Оценки']))
        print(self.dfc)
        

    def __ApplyFunction(y, funcWithPars):
        func = funcWithPars[1].value
        if(func == Functions.Power):       
            maxi = funcWithPars[2].value
            alpha = funcWithPars[3].value
            return func(vector=y, maximize = maxi, alpha=alpha)
        if(func == Functions.Logistic):
            maxi = funcWithPars[2].value
            beta = funcWithPars[3].value
            c = funcWithPars[4].value
            return func(vector=y, maximize=maxi, beta=beta, c=c)
        if(func == Functions.Bell):
            maxi = funcWithPars[2].value
            beta = funcWithPars[3].value
            c = funcWithPars[4].value
            return func(vector=y, beta=beta, c=c)

    def __GetWeights(wids):
        weights = []
        for i in wids.keys():
            weights.append(wids[i][5].value)
        return np.array(weights)


    def __Additive(weights, df):
        """Функция перемножает веса с значением критериев и выдает аддитивную скалярную оценку"""
        estimates = np.zeros(len(df))
        for i, col in enumerate(df.columns):    
            estimates+=df[col]*weights[i]
        return estimates
    
    def ShowContributionDiagram(self, column, obj):
        """
        Показать диаграммы полезности объекта obj по признаку column
        """
        weight = self.wids[column][5].value
        contribution = (self.dfc[column][obj] * weight) / self.dfc['Оценки'][obj] * 100
        
        norm = np.array((self.dfc[column]* weight) / self.dfc['Оценки'] * 100).mean()
        
        Show.ContributionDiagram(norm, contribution)
        plt.title('Вклад признака "' + str(column) + '" в оценку объекта ' + str(obj))
        
        
class UFonCriteria:
    """Построение ФП на основе критерия"""
    
    def __FindK(t, y_, y_a, y_b):
        return np.log(t)/np.log((y_ - y_a)/(y_b - y_a))
    
    def CreateFunctionOnCriteria(vector : np.array, n):
        """
        vector - наш показатель (с малым приращением)
        n -- на сколько частей разобьем
        Функция запрашивает предельное значение функции полезности для каждого отрезка и соответствующий этому значению объект (число из vector)
        Из этих данных находится k для каждого отрезка и строится кусочный график
        """
        ma = vector.max()
        mi = vector.min()
        diff = ma - mi
        step = np.round(diff / n)

        start = mi
        stop = mi + step
        ks =[]
        X = []
        Y = []
        while(stop <= ma):
            if (start == ma):
                break
            t = float(input(str.format('Укажите предельное значение полезности для отрезка [{0};{1}]', start, stop)))
            y_ = float(input(str.format('Укажите значение показателя, которое соответсвует указанному выше значению полезности для отрезка [{0};{1}]', start, stop)))
            k = UFonCriteria.__FindK(t, y_, start, stop)
            ks.append(k)
            interval = UFonCriteria.__Slice(vector, start, stop)
            X.extend(interval)
            Y.extend(Functions.Power(interval, alpha=k))
            start = stop
            stop = ma if stop + step >= ma else stop + step
        sns.lineplot(x=X, y=Y, linestyle='--')
        plt.grid()
        print(ks)
        
    def __Slice(vector, start, stop):      
        i = 0
        ret = []
        while (vector[i] < stop):
            ret.append(vector[i])
            i+=1
        return np.array(ret)


