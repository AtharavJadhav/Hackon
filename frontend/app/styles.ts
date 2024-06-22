import { StyleSheet } from "react-native";

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
  },
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  chatBox: {
    flex: 1,
    width: "100%",
    borderWidth: 1,
    borderColor: "#E3E3E3",
    padding: 10,
    marginBottom: 10,
    marginTop: 45,
    backgroundColor: "#E3E3E3",
    borderRadius: 8,
  },
  amazonImage: {
    width: 70,
    height: 70,
    marginTop: 300,
    borderRadius: 35,
    opacity: 0.9,
    alignSelf: "center",
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    marginBottom: 10,
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: "black",
    backgroundColor: "#E3E3E3",
    padding: 10,
    marginRight: 3,
    borderRadius: 15,
  },
  imageButton: {
    marginRight: 10,
  },
  previewImage: {
    width: 100,
    height: 100,
    marginTop: 10,
  },
  buyNowButton: {
    backgroundColor: "#4CAF50",
    padding: 10,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 5,
    marginTop: 10,
  },
  linksContainer: {
    marginTop: 10,
    maxHeight: 200, // Set a maximum height for the scrollable area
  },
  link: {
    color: "blue",
    textDecorationLine: "underline",
    marginVertical: 5,
  },
  backgroundImage: {
    flex: 1,
    width: "100%",
    height: "100%",
    justifyContent: "center",
    alignItems: "center",
  },
  startButton: {
    backgroundColor: "#FED813",
    width: 383,
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 7,
    top: 250,
  },
  startButtonText: {
    color: "black",
    fontSize: 18,
    fontWeight: "600",
    left: 83,
  },
  sendButtonImage: {
    width: 40,
    height: 40,
    right: -5,
  },
  uploadButtonImage: {
    width: 40,
    height: 40,
  },
  checkboxContainer: {
    alignContent: "center",
    flexDirection: "column",
    marginVertical: 10,
  },
  checkboxLabel: { fontSize: 12, marginRight: 10 },
});

export default styles;
