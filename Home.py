import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏡",
)

st.title("About Dataset")
st.text("""The datasets provided include the MLS player data for the Career Mode from FIFA 22 ("FIFA 22 MLS PLAYER RATINGS.csv"). The data allows multiple comparisons for the attributes of different MLS players in FIFA 22.

Some ideas of possible analysis:

Comparison between MLS clubs (what skill attributes define each club);
Which club has the highest average in each stat;
Player comparison for particular attribute/skill.""")

st.title("Content")
st.text("""Column Headers: PLAYER(player name), CLUB(club name), POS(position of a player), OVR(overall rating), PAC(pace rating), SHO(Shooting rating), PAS(Passing rating), DRI(Dribbling rating), DEF(Defence rating), PHY(Physicality rating).""")