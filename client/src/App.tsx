import Banner from "./components/Banner"
import InspectView from "./views/InspectView"
import UploadView from "./views/UploadView"

export default function App() {
    return (
        <div className="w-full max-w-xl p-4 h-screen mx-auto py-6 flex flex-col space-y-6">
            <Banner
                text="SPUTNIK"
                className="max-w-fit"
            />
            <InspectView />
            <UploadView />
        </div>
    )
}
