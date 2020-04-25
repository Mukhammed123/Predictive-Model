import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
import pandas_datareader as web
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, Rectangle
import model
class Main_Page(Screen):
    def __init__(self, **kwargs):
        super(Main_Page, self).__init__(**kwargs)
        self.data = web.get_nasdaq_symbols()
        self.symbols = self.data['NASDAQ Symbol'] # 8854
        self.security_name = self.data['Security Name']

        # Set orientation of the Box Layout
        self.box = BoxLayout(orientation='vertical', size_hint_y=None, height=Window.height, spacing=15)
        # self.box.pos_hint = {'x': 0,'y': 0}
        self.add_widget(self.box)
        self.flayout = FloatLayout()
        self.grid = GridLayout(cols=2)

        # Create and set up text input box
        self.text = TextInput(size_hint=(None, None), width='350dp', height='48dp')

        # Create and set up search button
        self.search = Button(text = 'search', size_hint=(None, None), width='60dp', height='48dp')
        self.search.bind(on_press=self.search_function)

        # Add input and button to 2 cols grid
        self.grid.add_widget(self.text)
        self.grid.add_widget(self.search)
        self.grid.pos_hint = {'x': 0, 'top': 9}
        self.flayout.add_widget(self.grid)
        # Add two cols grid to Box Layout
        self.box.add_widget(self.flayout)
        self.counter = 0

        self.arr_symbol = []
        self.arr_name = []
        self.arr_button = []
        for i in range(20):
            self.temp = GridLayout(cols=3)
            self.symbol_label = Label(text=self.symbols[i])
            self.symbol_label.font_size = 18
            self.symbol_label.size_hint_y = None
            self.symbol_label.size_hint_x = 0.08
            self.symbol_label.height = 19
            self.symbol_label.pos_hint = {'x': 0.1, 'top': 5}
            self.arr_symbol.append(self.symbol_label)
            # with self.symbol_label.canvas:
            #     Color(0, 1, 0, 0.25)
            #     Rectangle(pos=self.symbol_label.pos, size=self.symbol_label.size)
            self.flayout = FloatLayout()
            self.flayout.add_widget(self.symbol_label)
            self.temp.add_widget(self.flayout)

            self.name_label = Label(text=self.security_name[i])
            self.name_label.size_hint_y = None
            self.name_label.size_hint_x = 0.8
            self.name_label.font_size = 18
            self.name_label.height = 19
            self.name_label.pos_hint = {'x':0.1, 'top': 5}
            self.arr_name.append(self.name_label)
            self.flayout = FloatLayout()
            self.flayout.add_widget(self.name_label)
            # with self.flayout.canvas:
            #     Color(0, 1, 0, 0.25)
            #     Rectangle(pos=self.flayout.pos, size=self.flayout.size)
            #     print(self.flayout.size)
            self.temp.add_widget(self.flayout)
            #
            self.button = Button(size_hint=(None, None), text='Predict', width='60dp', height='40dp')
            self.button.id = self.symbols[i]
            self.button.bind(on_press=self.click)
            self.button.size_hint = (None,None)
            self.button.height = 25
            self.button.width = 70
            self.button.pos_hint = {'right': 1, 'top': 5}
            self.arr_button.append(self.button)
            self.flayout = FloatLayout()
            self.flayout.add_widget(self.button)
            self.temp.add_widget(self.flayout)
            #
            self.box.add_widget(self.temp)
        # self.scroll.add_widget(self.temp)
        # self.add_widget(self.box)
        self.arr_page = []
        self.current_page = 1
        self.prev_page = 1
        self.last = GridLayout(cols=30)
        # List of stocks
        for i in range(1, 31):
            self.page = Button(size_hint=(None, None), text='{}'.format(i), width='30dp', height='30dp')
            self.page.pos_hint = {'right': 1, 'bottom': 3}
            self.page.bind(on_press=self.page_function)
            self.arr_page.append(self.page)
            self.flayout = FloatLayout()
            self.flayout.add_widget(self.page)
            self.last.add_widget(self.flayout)
        self.arr_page[0].background_color = (0, 0, 1, 0.25)
        self.box.add_widget(self.last)
        self.arr_page_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        self.arr_page_numbers2 = [414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443]
        self.page_index_prev = 0
        self.page_index_cur = 0

    # When button Predict is clicked
    def click(self, object_button):
        info_class.label.text = object_button.id
        info_class.predict_button.id = object_button.id
        sm.current = 'info'
    # When search button is clicked
    def search_function(self, object_button):
        if self.text.text.upper() in self.symbols:
            search_class.l.text = self.text.text.upper()

            # print(self.data[self.text.text.upper()])
            search_class.button2.text = 'Predict'
            search_class.button2.id = search_class.l.text
            search_class.button2.bind(on_press=self.click)

            sm.current = "search"

        else:
            search_class.l.text = 'The stock is not found...'
            search_class.button2.text='Go Back'


    def go_back(self, button):
        sm.current = 'main'
    # Page listing
    def page_function(self, value):
        self.current_page = value.text

        if self.prev_page != self.current_page:
            if int(self.current_page) > 428:
                for i in range(30):
                    self.arr_page[i].text = str(self.arr_page_numbers2[i])
            elif int(self.current_page) > 15:

                for i in range(30):
                    self.arr_page[i].text = str(int(self.current_page) - 15 + i)
            else:
                for i in range(30):
                    self.arr_page[i].text = str(self.arr_page_numbers[i])

            for i in range(30):
                if self.arr_page[i].text == self.current_page:
                    self.page_index_cur = i
                self.arr_page[i].background_color = (1, 1, 1, 1)
            self.arr_page[self.page_index_cur].background_color = (0, 0, 1, 0.25)
            self.prev_page = self.current_page


            for i in range(20):
                if (int(self.current_page)-1)*20 + i >= 8854:
                    self.arr_symbol[i].text = self.symbols[i]
                    self.arr_name[i].text = self.security_name[i]
                    self.arr_button[i].id = self.symbols[i]
                else:
                    self.arr_symbol[i].text = self.symbols[(int(self.current_page)-1)*20 + i]
                    self.arr_name[i].text = self.security_name[(int(self.current_page)-1)*20 + i]
                    self.arr_button[i].id = self.symbols[(int(self.current_page)-1)*20 + i]
                if len(self.arr_name[i].text) > 170:
                    self.arr_name[i].font_size = 12
                elif len(self.arr_name[i].text) > 160:
                    self.arr_name[i].font_size = 14
                elif len(self.arr_name[i].text) > 145:
                    self.arr_name[i].font_size = 16
                elif len(self.arr_name[i].text) < 150 and self.arr_name[i].font_size < 18:
                    self.arr_name[i].font_size = 18

class Search_Page(Screen):
    def __init__(self, **kwargs):
        super(Search_Page, self).__init__(**kwargs)
        self.box = BoxLayout(orientation='vertical')
        self.button = Button(text='<-Back')
        self.button.bind(on_press=self.go_back)
        self.l = Label(text="none")
        self.box.add_widget(self.button)

        self.grid = GridLayout(cols=2)
        self.grid.add_widget(self.l)
        self.button2 = Button(text='Go Back')
        self.grid.add_widget(self.button2)
        self.box.add_widget(self.grid)
        self.add_widget(self.box)


    def go_back(self, button):
        sm.current = 'main'

class Info_Page(Screen):
    def __init__(self, **kwargs):
        super(Info_Page, self).__init__(**kwargs)
        self.box = BoxLayout(orientation='vertical')
        self.button = Button(text='<-Back')
        self.button.bind(on_press=self.go_back)
        self.button.size_hint = (0.2,0.2)
        self.box.add_widget(self.button)
        self.label = Label(text='Info Page')
        self.box.add_widget(self.label)

        self.predict_button = Button(text='Train & Predict')
        self.predict_button.bind(on_press=self.predict)
        self.box.add_widget(self.predict_button)
        self.add_widget(self.box)

    def go_back(self, button):
        sm.current = 'main'

    def predict(self, value):
        model.run_model(value.id)

info_class = Info_Page(name='info')
search_class = Search_Page(name='search')

sm = ScreenManager()
sm.add_widget(Main_Page(name='main'))
sm.add_widget(search_class)
sm.add_widget(info_class)

class Platform(App):
    def build(self):
        return sm


if __name__ == '__main__':
    Platform().run()