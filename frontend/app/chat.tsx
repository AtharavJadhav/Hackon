import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  Button,
  Image,
  TouchableOpacity,
  ScrollView,
  Switch,
} from "react-native";
//import CheckBox from "@react-native-community/checkbox";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";
import styles from "./styles";

export default function ChatScreen() {
  const [text, setText] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [response, setResponse] = useState("");
  const [objects, setObjects] = useState<string[]>([]);
  const [showLinks, setShowLinks] = useState(false);
  const [useHindi, setUseHindi] = useState(false);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImage(result.assets[0].uri);
    }
  };

  const sendRequest = async () => {
    if (!text || !image) {
      alert("Please provide both text and image");
      return;
    }

    const imageData = JSON.stringify({
      uri: image,
      type: "image/jpeg",
      name: "photo.jpg",
    });

    const formData = new FormData();
    formData.append("prompt", text);
    formData.append("image", imageData);
    formData.append("hindi", useHindi.toString()); // Append Hindi checkbox state

    try {
      const res = await axios.post(
        "http://localhost:8000/chat_with_image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Response from backend:", res.data); // Add this line
      setResponse(res.data.response);
      setObjects(res.data.objects);
      setShowLinks(false); // Reset showLinks when a new response is received
    } catch (error) {
      console.error("Error sending request:", error); // Add this line
      alert("Error sending request");
    }
  };

  const generateAmazonLinks = (object: string) => {
    const query = encodeURIComponent(object);
    return `https://www.amazon.in/s?k=${query}`;
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        {response && (
          <TouchableOpacity onPress={() => setShowLinks(!showLinks)}>
            <Image
              style={styles.sendButtonImage}
              source={require("../assets/images/shop.png")}
            />
          </TouchableOpacity>
        )}
        {showLinks && (
          <ScrollView style={styles.linksContainer}>
            {objects.map((object, index) => (
              <Text
                key={index}
                style={styles.link}
                onPress={() =>
                  window.open(generateAmazonLinks(object), "_blank")
                }
              >
                {object}
              </Text>
            ))}
          </ScrollView>
        )}
        <View style={styles.chatBox}>
          {response ? (
            <Text>{response}</Text>
          ) : (
            <Image
              style={styles.amazonImage}
              source={require("../assets/images/amazon.jpg")}
            />
          )}
        </View>
        <View style={styles.inputContainer}>
          <TouchableOpacity onPress={pickImage} style={styles.imageButton}>
            <Image
              style={styles.uploadButtonImage}
              source={require("../assets/images/pic.png")}
            />
          </TouchableOpacity>
          <View style={styles.checkboxContainer}>
            <Switch
              trackColor={{ false: "#767577", true: "#81b0ff" }}
              thumbColor={useHindi ? "#f5dd4b" : "#f4f3f4"}
              ios_backgroundColor="#3e3e3e"
              value={useHindi}
              onValueChange={setUseHindi}
            />
            <Text style={styles.checkboxLabel}>Use Hindi</Text>
          </View>
          <TextInput
            style={styles.textInput}
            placeholder="What's on ur mind? "
            value={text}
            onChangeText={setText}
          />
          <TouchableOpacity onPress={sendRequest} style={styles.imageButton}>
            <Image
              style={styles.sendButtonImage}
              source={require("../assets/images/send2.png")}
            />
          </TouchableOpacity>
        </View>
        {image && <Image source={{ uri: image }} style={styles.previewImage} />}
      </View>
    </ScrollView>
  );
}
