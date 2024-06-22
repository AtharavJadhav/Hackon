import {
  Text,
  View,
  Button,
  ImageBackground,
  TouchableOpacity,
} from "react-native";
import { useRouter } from "expo-router";
import styles from "./styles";

export default function Index() {
  const router = useRouter();

  return (
    <ImageBackground
      source={require("../assets/images/bg1.jpeg")}
      style={styles.backgroundImage}
    >
      <View style={styles.container}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={() => router.push("/chat")}
        >
          <Text style={styles.startButtonText}>#DoItWithAmazon</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
}
