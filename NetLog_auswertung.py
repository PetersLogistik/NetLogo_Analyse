# -*- UTF-8 -*-
"""
@Autor: Patick Peters
@Date: 18.03.2023
"""
__version__ = 1.0

class netlog_analyse():
    #Import
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
    from tqdm import tqdm # für Statusleisten
    
    # def's
    def __init__(self, url, agents):
        # Globales
        global url
        global agents
        self.url = r".\NetLogo_Analyse"
        self.agents = ( 2, 3, 4, 5, 10, 15, 20, 25, 35, 45)
        
    def save_beckup_fundagl(self, jpv):
        # das eingegebene Array wird in dieser Funktion gespeichert. Da nur ein bestimmtes Array gespeichert wird ist die Funktion starr vom "header". 
        f = r"\csv_daten\fundagl.csv"
        head = ["agenten", "fundermentalgleichung", "dichte", "v_quer"]
        jpv = self.pd.DataFrame.from_records(jpv) # In schleife gezogen um auf den aktuellen bei Abbruch zurück zu kommen. 
        jpv.to_csv(self.url + f, header=head, index=False) # save & überschreibend

    def load_beckup(self):
        # Diese Funktion läd eine bestimmte zuvor gespeicherte Datei.
        f = r"\csv_daten\fundagl.csv"
        jpv = self.pd.read_csv(self, self.url+f, header=0)

        return jpv

    """
        Zu Aufgabe 1 
    """
    def einlesen(self, agent, dat = r'\N-', col = False, s = False):
        # Im folgendem werden die CSV-Dateinen als Dataframe (df) eingelesen und 
        # zum einen ans 'Plotten' weiter und zurück gegeben. Darüber hinaus wird 
        # die erste Spalte unbenannt. Standart mäßig wird eine Datei im Format "\N-"+Zahl eingelesen. 
        # Wenn col auf True gesetzt wird und ein anderer Beginn zb. "\dfv_"+Zahl eingegtragen wird, 
        # dann wird anschließend eine andere Plot Funktion aufgerufen.
        f = r'\csv_daten'+ dat + str(agent)+r'.csv'
        df = self.pd.read_csv(self.url + f, sep=",", header=0)
        df.rename(columns = {'# time': 'time'}, inplace = True)
        if col == True:
            self.col_plot(df, agent, s)
        else:
            self.plotten(df, agent, s)
        return df
        
    def plotten(self,df, agent, s = False):
        # Als erstes wir der Savename & Title des Plots festgelegt. Anschließend 
        # werden die ersten 100 sek aus dem DF genommen (=> vgl. NetLogo), 
        # geplottet und gespeichert. 
        f = r'\\'+str(agent)+r'_trajec.png' # saveort
        title = 'Trajectories for ' + str(agent) + ' Agents' 
        
        df_100 = df[df['time'] < 50 + 300] # abweichung auf 50, da es sonst unübersichtlich wird.
        ax = df_100.plot(kind = 'scatter', x = 'x', y = 'time', title = title, figsize=(10,15)) 
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        
        if s == True: # Das 'show' wird zu testzwecken Aktiviert.
            self.plt.show()
        else:
            self.plt.savefig(self.url+f)
            print('\n', 'Trajectorien gespeichert.')

    """ 
        Zur Aufgabe 2 
    """
    def geschwindigkeit(self, df, agent):
        # Gl.3: v(t) = ( x(t+dt)-x(t) ) / dt mit dt = Differenz zweier aufeinander 
        #       folgender Zeitschritte, x: Standort des Fahrzeugs
        # Nach der angegebenen Gleichung wird die Geschwindigkeit in t berechnet und ans df gehängt. 
        # Abschießend wird das Df mit der neuen Spalte gespeichert.
        f = r'\csv_daten\dfv_'+str(agent)+r'.csv'
        head = ["time", "id", "x", "vi"]
        
        for id in range(agent):
            v = []
            i = 0
            
            df_a = df[df['id'] == id].copy()
            df_a.sort_values('time')
            
            for i in self.tqdm(range(len(df_a)-1), desc = f'Durchlauf {id +1} von {agent}'): # Statusbar
                z = self.np.sqrt((df_a['x'].iloc[i+1] - df_a['x'].iloc[i])**2) # Distanzberechnung
                n = (df_a['time'].iloc[i+1] - df_a['time'].iloc[i]) # Zeitschrittsrechnung
                vt_neu = z / n # Geschwindigkeit
                if vt_neu < 5: # Austrittsgeschwindigkiet = Eintrittsgeschwindigkeit -> überspringen der Berechnung # solange vt < 5 m/s, da Ziel vt max 4m/s ist.
                    vt = vt_neu
                v.append(vt)
            
            v.append(vt) # Es wird angenommen, dass die Geschwindigkeit sich im letzten Punkt nicht, wie beim Aus und Eintritt ins System
            df_a['v'] = v
            if id == 0 : # Für die erste Id muss die v-Spalte erzeugt werden.
                df = df.merge(df_a, how = 'left')
            else: # Da die v-Spalte bereits vorhanden ist und 'NaN' in den noch nicht gefüllten ID-Zellen steht, muss nur noch das Uself.pdate laufen.
                df.uself.pdate(df_a, join='left', overwrite=False, filter_func=None, errors='ignore')
            
        df.to_csv(self.url + f, header=head, index=False) # save dfv
            
        return df
            
    def mittlere_geschwindigkeit(self, df, agent):
        # Gl.4: v = 1/N*Nt sum(0-N: sum(tstart-tende: v(t))) mit n: Agenten, v: Geschwindigkeit, nt: Zeitschritte,
        # Nach der angegebenen Gleichung wird die mittlere Geschwindigkeit berechnet und für die momentane Betrachtung ausgegeben.
        # 1 / 2 * 385634 :== 1/gesammte Länge # sum(0-N: sum(tstart-tende: v(t))) := Summe aller v(t)
        vq = df['v'].mean()
        print('Agenteanzahl:', agent, 'mit v[quer] =', vq, 'm/s')
        
        return vq
        
    def dichte_gesammt(self, agent):
        # Gl.5: p = N / L mit L: Länge (m), N: Angentenanzahl
        # Nach der angegebenen Gleichung wird die Dichte im System berechnet.
        l = 52 # aus Aufgabenstellung
        p = agent / l
        
        return p

    def funda_gleichung(self, df, agent):
        # Gl.1: j = ρ · v mit p: Dichte, v: Geschwindigkeit
        # Nach der angegebenen Gleichung wird das J berechnet.
        # Zuvor werden die dafür benötigten Werte in den entsprechenden Funktionen berechnet.
        # Abschließend wird die entstandene Liste zurück gegeben für weitere Rechnungen bzw. Plots.
        df_v = self.geschwindigkeit(df, agent)
        p = self.dichte_gesammt(agent)
        vq = self.mittlere_geschwindigkeit(df_v, agent)
        
        j = p * vq
        funda = [agent, j, p, vq]
            
        return funda

    def funda_plot(self, pv, s = False):
        # Diese Funktion plottet das Fundamentaldiagramm und speichert es ab.
        f = r"\\fundamentaldiagramm.png"
        title = 'Fundamentaldiagramm'
        
        ax = pv.plot(kind = 'scatter', x = 'dichte', y = 'v_quer', title = title, figsize=(10,15))
        ax.set_xlabel("Dichte (p)")
        ax.set_ylabel("Geschwindigkeit (v) in [m/s]")
        
        if s == True: # Das 'show' wird zu testzwecken Aktiviert.
            self.plt.show()
        else:
            self.plt.savefig(self.url+f)
        
    """ 
        Zu Aufgabe 3 
    """
    def funct(self, p, a, b, c):
        # Siehe: https://firedynamics.github.io/LectureComputerScience/content/Script/03_datenanalyse/035_optimierung.html?highlight=self.curve_fit
        # v[p] == Piecewise[{{a[1/c - 1/b], p <= c}, {a[1/p - 1/b], p > c}}] :: lt. Wolfram
        def top(p):
            return a*(1/c - 1/b)

        def low(p):
            return a*(1/p - 1/b)
        
        return self.np.piecewise(p, [p <= c, p > c], [top, low])

    def funda_plot2(self, pv, s = False):
        # Diese Funktion fittet das Fundamentaldiagramm. 
        # Zu näcshte wird das eine Kurve gefittet. Die Formel wird in der Def. oberhalb angeführt. 
        # Anschließend wird die Kurve und die Fundamentalpunkte geplottet.
        f = r"\\self.curve_fit.png"
        p0 = [3, 0.8, 2]
        
        popt, pcov = self.curve_fit(f = self.funct, xdata = pv['dichte'].to_numpy(), ydata = pv['v_quer'], p0 = p0)
        x_fit = self.np.linspace(min(pv['dichte']), max(pv['dichte']), 100)
        
        self.plt.figure(figsize=(10,15))
        
        self.plt.plot(pv['dichte'], pv['v_quer'], "bo", label = 'Fundamentaldiagramm')
        self.plt.plot(x_fit, self.funct(x_fit, *popt), label = f'Initial Parameter: a = {p0[0]}, b, = {p0[1]}, c={p0[2]}', c = 'r')
        
        
        self.plt.xlabel("Dichte (p)")
        self.plt.ylabel("Geschwindigkeit (v) in [m/s]")
        self.plt.title("Curve-Fit des Fundamentaldiagramms")
        self.plt.legend()
        self.plt.grid(True)
        
        if s == True: # Das 'show' wird zu testzwecken Aktiviert.
            self.plt.show()
        else:
            self.plt.savefig(self.url+f)

    """
        Zu Aufgabe 4
    """
    def lok_dichte(self, df, a = 'N0N'):
        # Gl.6: pi = 1 / di mit di: siehe Abbildung 3
        # Jene abgelidete Rechnung findet in dieser Funktion statt. Es wird ein df mit 'x' eingegeben. Es durchläuft die Berechnung. 
        # Diese wird in dem unterhab aufgezeitem Bild erläutert. Die If-Abfragen werden Bildlich und Rechnerisch dargestellt. 
        # Abschließend wird die Liste mit den lokalen Dichten zurück gegeben. Der Erste und letze Wert ist immer -999 da für diese keine lokale Dichte errechnet werden kann.
        p = [-999]
        
        for i in self.tqdm(range(1, len(df) - 1), desc = f'Dichte für {a} Agenten' ):
            h1 = df['x'].iloc[i-1] 
            h2 = df['x'].iloc[i]
            h3 = df['x'].iloc[i+1]
            
            if h2 < h3 < h1: # Bruch zw h1 & h2
                di = ((26 - h1) + (26 + h2)) + self.np.sqrt((h3 - h2)**2)
            elif h3 < h1 < h2: # Bruch zw h2 & h3
                di = self.np.sqrt((h2 - h1)**2) + ((26 - h2) + (26 + h3))
            else:
                di = (self.np.sqrt((h2 - h1)**2) + self.np.sqrt((h2 - h1)**2))
            
            pi = 1 / (di / 2)
            p.append(pi)
            
        p.append(-999)
        
        return pi
        
    def individuelle_funda(self):
        # Diese Definition ist der lok_dichte vorangestellt. Diese offnet, sortiert nach Zeit und x-Werten, 
        # übergibt die dfs an lok_dichte und speichert abschließend das Ergebnis.
        for agent in self.agents:
            
            head = ["time", "id", "x", "vi", "pi"]
            
            df = self.pd.read_csv(self.url + f'\csv_daten\dfv_{agent}.csv', header=0)
            df = df.sort_values(by = ['time', 'x'], ignore_index = True)
                
            pi = self.lok_dichte(df, agent)

            df['pi'] = pi
            df.to_csv(self.url + f'\csv_daten\dfp_{agent}.csv', header=head, index=False) # save df
            
    """
        Zu Aufgabe 5
    """
    def funda_plot_3(self, pv, s = False):
        # Diese Funktion stellt einen neuen Plot mit zwei Kurven, der einen, die zuvor bereits erstellt wurde und 
        # einer, die sich auf die lokale dichte Stützt. Nr.3 ist der Fundamentalplot vom Eingang.
        # Dazu wird der eingegebene Dateframe aufbereitet ( neu zusammen geschrieben ins Format Geschwindigkeit ~ Dichte ) und die -999 heraus genommen. 
        # Anschließend wir der bereitserstellten Fangfolge nach Fundamentalpunkteu nd Curve_fit_1 vorran, dann der neue Plot erstellt. Betittelt mit "Werte aus Aufgabe 4"
        df_b = 0
        p0 = [5,0.5,0.1]
        f = r'\self.curve_fit_vgl.png'
        
        for n in self.tqdm(range(len(self.agents)), desc = 'Dataframe aufbereiten'): # Werte aus Aufgabe 4 aufbereiten
            df = self.pd.read_csv(self.url + f'\csv_daten\dfp_{self.agents[n]}.csv', header=0)
            df = df[['vi','pi']]
            
            if n == 0:
                df_b = df
            else:
                df_b = self.pd.concat([df_b, df], ignore_index=True)
                
        df_b = df_b[df_b['pi'] != -999]
        
        self.plt.figure(figsize=(10,15))  
        self.plt.plot(pv['dichte'], pv['v_quer'], "bo", label = 'Fundamentaldiagramm') # Fundamentaldiagramm

        popt_2, pcov_2 = self.curve_fit(self.funct, xdata = pv['dichte'].to_numpy(), ydata = pv['v_quer'], p0 = p0) # Werte aus Aufgabe 2 
        x_fit_2 = self.np.linspace(min(pv['dichte']), max(pv['dichte']), 100)
        self.plt.plot(x_fit_2, self.funct(x_fit_2, *popt_2), label = f'Werte aus A2. Initial Parameter: a = {p0[0]}, b, = {p0[1]}, c={p0[2]}', c = "g")
        
        popt_4, pcov_4 = self.curve_fit(self.funct, xdata = df_b['pi'].to_numpy(), ydata = df_b['vi'], p0 = p0) # Werte aus Aufgabe 4
        x_fit_4 = self.np.linspace(min(df_b['pi']), max(df_b['pi']), 100)
        self.plt.plot(x_fit_4, self.funct(x_fit_4, *popt_4), label = f'Werte aus A4. Initial Parameter: a = {p0[0]}, b, = {p0[1]}, c={p0[2]}', c = "r")
        
        self.plt.xlabel('Dichte (p)')
        self.plt.ylabel('Geschwindigkeit (v) in [m/s]')
        self.plt.title("Curve-Fit Vergleich im Fundamentaldiagramm")
        self.plt.legend()
        self.plt.grid(True)
        
        if s == True: # Das 'show' wird zu testzwecken Aktiviert.
            self.plt.show()
        else:
            self.plt.savefig(self.url+f)
        
        self.fluss(df_b)
        
    """
        Zu Aufgabe 6
    """ 
    def fluss(self, df):
        # j = p * vq
        # Nach der gegebenen Gleichung wird j berechnet und an das Df gehängt sowie anschließend gespeichter.
        f = r"\csv_daten\df_vp.csv"
        
        ji = df['pi'] * df['vi']
        df['j'] = ji
        
        df.to_csv(self.url+f, header = ['vi', 'pi', 'j'], index = False)
        
    def funda_fluss_plot(self, jpv, s = False):
        # Diese Funktion öffnet die zuvor erstllte CSV Datei mit dem errechnetem j. Die in Aufgabe 2 erstellte Liste wird 
        # eingegeben und erhält den selben vorgang wie die nun geladene df.
        # Als erstes wird die Zeile des Max j heraus gesucht. Anschließend wird ein Plot über alle gegeben Werte erstellt.
        # Anschließend wird eine Vertikale Linie z beiden Maxima erstellt. -> Werte gegeben über die Max-Zeilen.
        # Abschließend werden die Koordinaten der jewailigen kritischen Dichten an die Vertikale Linie geschrieben.
        f = r'\csv_daten\df_vp.csv'
        
        df = self.pd.read_csv(self.url+f, header=0)
        df = df[['pi', 'j']].groupby(df['pi']).mean()
        
        
        max2 = jpv.loc[jpv['fundermentalgleichung'].idxmax()]
        max4 = df.loc[df['j'].idxmax()]
        
        self.plt.figure(figsize=(10,15))
        
        self.plt.plot(df['pi'], df['j'], "bo-.", label='aus Aufgabe 4') # , title = title, figsize=(10,15))
        self.plt.plot(jpv['dichte'], jpv['fundermentalgleichung'], "ro-.", label='aus Aufgebe 2')

        self.plt.vlines(max4['pi'], ymin=0, ymax=max4['j'], color='blue', alpha=0.5, label='max Kapazität aus Aufgabe 4') # max linie
        self.plt.vlines(max2['dichte'], ymin=0, ymax=max2['fundermentalgleichung'], color='red', alpha=0.5, label='max Kapazität aus Aufgabe 2') # max linie
        
        self.plt.annotate(f"pk = ({round(max4['pi'], 3)}, {round(max4['j'], 3)})", (max4['pi'], 0.02))
        self.plt.annotate(f"pk = ({round(max2['dichte'], 3)}, {round(max2['fundermentalgleichung'], 3)})", (max2['dichte'], 0.0))
        
        self.plt.xlabel('Dichte (p)')
        self.plt.ylabel('Fundermentwert')
        self.plt.title("Fundamentaldiagramm")
        self.plt.legend()
        self.plt.grid(True)
        
        if s == True: # Das 'show' wird zu testzwecken Aktiviert.
            self.plt.show()
        else:
            self.plt.savefig(self.url+ r'\funda_vgl.png')

    """
        Zu Aufgabe 7:
    """ 
    def fulss_kapa(self, jpv):
        # Gl.2: C = max(j) 
        # Nach der angegebenen Gleichung wird das Maximum zurück gegeben.
        # Da dies nur eine Ausgabe in der Console erzeugt und diese Aufgabe bereits unter 
        # Punkt 6 mit eingebunden wurde ist diese Funktion eigentlich überflüssig. Aber wird 
        # für die Erörterung des Auftrags 7 stehengelassen.
        f = r'\csv_daten\df_vp.csv'
        
        df = self.pd.read_csv(self.url+f, header=0)
        print('Max J in Aufgabe 4 ist:', df['j'].max())
        
        mj = max(jpv['fundermentalgleichung'])
        print('Max J in Aufgabe 2 ist:', mj)
        
    """
        Zu Aufgabe 8
    """     
    def col_plot(self, df, agent, s = False):
        # Als erstes wir der Savename & Title des Plots festgelegt. Anschließend 
        # werden die ersten 100 sek aus dem DF genommen (=> vgl. NetLogo), 
        # Nach geschwindigkeiten gefiltert, und in entsprechende Arrays gepackt, werden sie abschließend geplotet.
        # 
        f = r'\\'+str(agent)+r'_col_trajec.png'
        
        df_100 = df[df['time'] < 50 + 300]

        df_b = df_100[df_100['vi'] < 0.5]
        df_r = df_100[(df_100['vi'] < 1.5) & (df_100['vi'] >= 0.5)]
        df_m = df_100[(df_100['vi'] < 2.5) & (df_100['vi'] >= 1.5)]
        df_y = df_100[(df_100['vi'] < 3.5) & (df_100['vi'] >= 2.5)]
        df_g = df_100[df_100['vi'] >= 3.5]
        
        self.plt.figure(figsize=(10,15))
        
        self.plt.plot(df_g['x'], df_g['time'], 'go', label='Freier Fuss mit >= 3,5 m/s')
        self.plt.plot(df_y['x'], df_y['time'], 'yo', label='Übergangs Fuss mit >= 2,5 m/s & < 3,5 m/s')
        self.plt.plot(df_m['x'], df_m['time'], 'mo', label='Stockender Fuss mit >= 1,5 m/s < 2,5 m/s')
        self.plt.plot(df_r['x'], df_r['time'], 'ro', label='Stockender Fuss mit >= 0,5 m/s < 1,5 m/s')
        self.plt.plot(df_b['x'], df_b['time'], 'ko', label='Stockender Fuss mit < 0,5 m/s')

        self.plt.xlabel('Abschnitt (x)')
        self.plt.ylabel('Zeit (t)')
        self.plt.title(f"Trajectorien mit {agent} Agenten")
        self.plt.legend(loc='upper left')
        self.plt.grid(True)
        
        if s == True: # Das 'show' wird zu testzwecken Aktiviert.
            self.plt.show()
        else:
            self.plt.savefig(self.url+f)
            print('\n', 'colore Trajectorien gespeichert.')

        
    """
        Ablauf
    """
    def durchlauf1(self, agent):
        # Aufgabenteil 1 bis 2
        jpv =[] # liste von der Agenten, Fundermentalgleichung (j), Dichte (p) und mittlerer Geschwindigkeit (vq)
        for a in agent:
            df = self.einlesen(a)
            j = self.funda_gleichung(df, a)
            jpv.append(j)
            self.save_beckup_fundagl(jpv)
            
    def durchlauf2(self):
        # Aufgabenteil 2 bis 5
        jpv = self.load_beckup()
        self.funda_plot(jpv)
        self.funda_plot2(jpv)
        self.individuelle_funda()

    def durchlauf3(self):
        # Aufgabenteil 6 bis 8
        jpv = self.load_beckup()
        self.funda_plot_3(jpv)
        self.funda_fluss_plot(jpv)
        self.fulss_kapa(jpv)

    def durchlauf4(self, agent):
        for a in agent:
            self.einlesen(a, dat = r'\dfv_', col = True)

    """
        Test
    """
    def test_fluss(self):
        jpv = self.load_beckup()
        self.funda_fluss_plot(jpv, s = True)
        # fulss_kapa(jpv)

    def desingn(self):
        # Zum Diagramme anzeigenlassen für die Parameter Aself.npassung ! Ohne Speicherung!
        s = True
        for a in self.agents:
            self.einlesen(a, s = s)
        jpv = self.load_beckup()
        self.funda_plot(jpv, s)
        self.funda_plot2(jpv, s)
        self.funda_plot_3(jpv, s)
        self.funda_fluss_plot(jpv, s)
        for a in self.agents:
            self.einlesen(a, dat = r'\dfv_', col = True, s = s)
        
    def test_v_quer(self, agent = 2):
        df = self.pd.read_csv(self.url + f'\csv_daten\dfv_{agent}.csv', header=0)
        self.mittlere_geschwindigkeit(df, agent)
        
    def test_individuelle_funda(self, agent = 2):
        self.individuelle_funda()

    def test_funda_plot_3(self):
        self.funda_plot_3(self.load_beckup())

def kompletter_durchlauf(agents):
    netlog_analyse.durchlauf1(agents)
    netlog_analyse.durchlauf2()
    netlog_analyse.durchlauf3()
    netlog_analyse.durchlauf4(agents)

if __name__ == "__main__":
    import sys
    a = ( 2, 3, 4, 5, 10, 15, 20, 25, 35, 45)
    kompletter_durchlauf(a)