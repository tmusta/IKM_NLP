import tkinter as tk

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


class GUI:
  def __init__(self):
    self.window = tk.Tk()
    self.window.title("Disambiguation is a Difficult Word")
    self.window.configure(background=primary_color)


    #word entry field init
    self.word_string = tk.Entry(self.window, fg=secondary_text_color, background=secondary_color, insertbackground=primary_text_color, width=50, font=normal_font)
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

    #guess label init
    self.results_guess_label = tk.Label(self.window, text="", fg=primary_text_color, background=primary_color, font=middle_font_bold)
    self.results_guess_label.grid(column=1, row=6, sticky="W", pady=10, padx=5, columnspan=2)

    #definition label init
    self.results_definition_label = tk.Label(self.window, text="", fg=primary_text_color, background=primary_color, font=normal_font)
    self.results_definition_label.grid(column=1, row=7, sticky="W", pady=10, padx=5, columnspan=2)

    #example sentance label init
    self.results_example_label = tk.Label(self.window, text="", fg=primary_text_color, background=primary_color, font=normal_font)
    self.results_example_label.grid(column=1, row=8, sticky="W", pady=10, padx=5, columnspan=2)


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

    #reset placeholders
    self.word_string.insert(0, "Word to Disambiguate")
    self.word_string.configure(fg=secondary_text_color)
    self.context_string.insert("1.0", "Context Sentance for the Word")
    self.context_string.configure(fg=secondary_text_color)

    self.results_header_label.configure(text="Waiting for Inputs")
    self.results_guess_label.configure(text="")
    self.results_definition_label.configure(text="")
    self.results_example_label.configure(text="")

  def disambiguate(self, event):
    #Event handler for the disambiguate button, currently results hard coded. Will update after other parts ready.
    guess_string, definition_string, example_string = "Hard (adjective)", "not easy; requiring great physical or mental effort to accomplish or comprehend or endure", "\"why is it so hard for you to keep a secret?\""
    #TODO connect the above to the classifier
    print(event)
    self.results_header_label.configure(text="Results:")
    self.results_guess_label.configure(text=guess_string)
    self.results_definition_label.configure(text=definition_string)
    self.results_example_label.configure(text=example_string)



if __name__ == "__main__":
  #create GUI object
  gui = GUI()

  #call the window main loop
  gui.window.mainloop()
