import ASCIIArt from "./components/ASCIIArt"
import JSONForm from "./components/JSONForm"
import Loading from "./components/Loading"
import { HexColors } from "./lib/colors"

const ascii = [
String.raw`.______.....______...__..__.....______...__...__.....__.....__..__....`,
String.raw`/\..___\.../\..==.\./\.\/\.\.../\__.._\./\."-.\.\.../\.\.../\.\/./....`,
String.raw`\.\___..\..\.\.._-/.\.\.\_\.\..\/_/\.\/.\.\.\-...\..\.\.\..\.\.._"-...`,
String.raw`.\/\_____\..\.\_\....\.\_____\....\.\_\..\.\_\\"\_\..\.\_\..\.\_\.\_\.`,
String.raw`..\/_____/...\/_/.....\/_____/.....\/_/...\/_/.\/_/...\/_/...\/_/\/_/.`,
]

export default function App() {
    return (
        <div className="max-w-[80ch] h-screen mx-auto py-6 flex flex-col items-center">
            <ASCIIArt art={ascii} bgPattern={new RegExp(/\./g)} bgColor={HexColors.BLACK} />
            <JSONForm />
            <Loading color={HexColors.BLUE} size={20} />
        </div>
    )
}
