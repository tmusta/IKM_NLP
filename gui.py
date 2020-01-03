import tkinter as tk
from lesk import extended_lesk, standard_lesk

primary_color = "#161616"
secondary_color = "#1b1c1d"
primary_text_color = "#ffffff"
secondary_text_color = "#a0a0a0"

accent_color = "#383838"
effect_color = "cecece"

font = "Sans Serif"

header_font = ("Sans Serif", "24")
middle_font_bold = ("Sans Serif", "18", "bold")
normal_font = ("Sans Serif", "12")

class ResultsFrame:
  def __init__(self, parent_window, title, column, row):
    self.window = tk.Frame(parent_window, background=secondary_color)
    self.window.grid(column=column, row=row, sticky="W,E", pady=10, padx=5, columnspan=1)
    self.window.grid_remove()

    
    self.title_label = tk.Label(self.window, text=title, fg=primary_text_color, background=secondary_color, font=middle_font_bold, wraplengt=400, justify="left")
    self.title_label.grid(column=1, row=1, sticky="W,E", pady=10, padx=5, columnspan=1)

    #guess label init
    self.guess_label = tk.Label(self.window, text="", fg=primary_text_color, background=secondary_color, font=middle_font_bold, wraplengt=400, justify="left")
    self.guess_label.grid(column=1, row=2, sticky="W", pady=10, padx=5, columnspan=1)

    #definition label init
    self.definition_label = tk.Label(self.window, text="", fg=primary_text_color, background=secondary_color, font=normal_font, wraplengt=400, justify="left")
    self.definition_label.grid(column=1, row=3, sticky="W", pady=10, padx=5, columnspan=1)

    #example sentance label init
    self.example_label = tk.Label(self.window, text="", fg=primary_text_color, background=secondary_color, font=normal_font, wraplengt=400, justify="left")
    self.example_label.grid(column=1, row=4, sticky="W", pady=10, padx=5, columnspan=1)

    self.visibility = False

  def reset(self):
    self.guess_label.configure(text="")
    self.definition_label.configure(text="")
    self.example_label.configure(text="")
    if self.visibility:
      self.toggle_visibility()

  def toggle_visibility(self):
    if self.visibility:
      self.window.grid_remove()
      self.visibility = False
    else:
      self.window.grid()
      self.visibility = True

class GUI:
  def __init__(self):
    self.window = tk.Tk()
    self.window.title("Disambiguation is a Difficult Word")
    self.window.configure(background=primary_color)


    #word entry field init
    self.word_string = tk.Entry(self.window, fg=secondary_text_color, background=secondary_color, insertbackground=primary_text_color, font=normal_font)
    self.word_string.insert(0, "Word to Disambiguate")
    self.word_string.bind("<FocusIn>", self.focus_in_word)
    self.word_string.bind("<FocusOut>", self.focus_out_word)
    self.word_string.grid(padx=5, pady=10, column=1, row=1, sticky="W,E")

    #context text field init
    self.context_string = tk.Text(self.window, fg=secondary_text_color, background=secondary_color, insertbackground=primary_text_color, height=3, font=normal_font)
    self.context_string.insert("1.0", "Context Sentance for the Word")
    self.context_string.bind("<FocusIn>", self.focus_in_sentance)
    self.context_string.bind("<FocusOut>", self.focus_out_sentance)
    self.context_string.grid(padx=5, pady=10, column=1, row=2, sticky="W,E")

    #disambiguate button init
    self.button_disambiguate = tk.Button(self.window, bg=secondary_color, fg=primary_text_color, text="Disambiguate!", activebackground=accent_color, activeforeground=primary_text_color, font=normal_font) 
    self.button_disambiguate.grid(column=1, row=4, columnspan=1, sticky="W", ipady=5, padx=5)
    self.button_disambiguate.bind("<Button-1>", self.disambiguate)

    #reset button init
    self.button_reset = tk.Button(self.window, bg=secondary_color, fg=primary_text_color, text="Reset", activebackground=primary_color, activeforeground=primary_text_color, font=normal_font)
    self.button_reset.grid(column=1, row=4, columnspan=1, sticky="E", ipady=5, padx=5)
    self.button_reset.bind("<Button-1>", self.reset)

    #results header init
    self.results_header_label = tk.Label(self.window, text="Waiting for Inputs", fg=primary_text_color, background=primary_color, font=header_font)
    self.results_header_label.grid(column=1, row=5, sticky="W", pady=10, padx=5)
    
    #results frame
    self.results_parent = tk.Frame(self.window, background=primary_color)
    self.results_parent.grid(column=1, row=6, sticky="W,E", pady=10, padx=5, columnspan=2)

    self.lesk = ResultsFrame(self.results_parent, "Standard Lesk", column=1, row=1)
    self.ex_lesk = ResultsFrame(self.results_parent, "Extended Lesk", column=2, row=1)


    

  def focus_in_word(self, event):
    #focus in event handler for word field
    #quicker to do dedicated functions rather than general one since there is only one Entry and Text element each.
    if self.word_string.get() == "Word to Disambiguate":
      self.word_string.delete(0, tk.END)
      self.word_string.configure(fg=primary_text_color)

  def focus_out_word(self, event):
    #focus out event handler for word field
    if self.word_string.get() == "":
      self.word_string.insert(0, "Word to Disambiguate")
      self.word_string.configure(fg=secondary_text_color)

  def focus_in_sentance(self, event):
    #focus in event handler for word field
    if self.context_string.get("1.0", tk.END) == "Context Sentance for the Word\n":
      self.context_string.delete("1.0", tk.END)
      self.context_string.configure(fg=primary_text_color)

  def focus_out_sentance(self, event):
    #focus out event handler for context field
    if self.context_string.get("1.0", tk.END) == "\n":
      self.context_string.insert("1.0", "Context Sentance for the Word")
      self.context_string.configure(fg=secondary_text_color)

  def reset(self, event):
    #reset all fields and results
    print("reset event", event)
    self.word_string.delete(0,tk.END)
    self.context_string.delete("1.0", tk.END)
    self.window.focus()
    self.results_header_label.configure(text="Waiting for Inputs")

    #reset result_frames
    self.lesk.reset()
    self.ex_lesk.reset()
    

  def extend_tag(self, tag):
    tag_table = {"a": "Adjective", "n": "Noun", "v":"Verb", "s": "Adjective Satellite", "r": "Adverb"}
    try:
      return tag_table[tag]
    except KeyError:
      return tag

  def disambiguate_and_insert_results(self, frame, word, sentance, func):
    frame.reset()
    try:
      result = func(word,sentance)
      if result is None:
        frame.guess_label.configure(text="No Results Available")
        if(frame.visibility == False):
          frame.toggle_visibility()
        return
          
        
      try:
        definition = result.definition()
      except AttributeError:
        definition = "No Definition Available."

      try:
        example = result.examples()[0]
      except IndexError:
        example = "No Example Sentance Available."
      frame.guess_label.configure(text=result.name() + " (" + self.extend_tag(result.pos())+")")
      frame.definition_label.configure(text=definition)
      frame.example_label.configure(text=example)
      if(frame.visibility == False):
        frame.toggle_visibility()
      self.results_header_label.configure(text="Results:")
    except ValueError or AttributeError:
      self.results_header_label.configure(text="Invalid Parameters.")


  def disambiguate(self, event):

    self.results_header_label.configure(text="Processing...")
    
    word = self.word_string.get()
    sentance = self.context_string.get("1.0", tk.END)
    if not word.lower() in sentance.lower():
      self.results_header_label.configure(text="Sentance doesn't contain the word.")
      return
    #do standard lesk results
    self.disambiguate_and_insert_results(self.lesk,word,sentance,standard_lesk)
    #do extended lesk results
    self.disambiguate_and_insert_results(self.ex_lesk,word,sentance,extended_lesk)
      



if __name__ == "__main__":
  #create GUI object
  gui = GUI()

  #call the window main loop
  gui.window.mainloop()
