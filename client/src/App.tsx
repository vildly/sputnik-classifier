import Banner from "./components/Banner"
import InspectView from "./views/InspectView"
import UploadView from "./views/UploadView"

export default function App() {
    return (
        <div className="w-full max-w-lg h-screen mx-auto py-2 flex flex-col space-y-6">
            <Banner
                text="SPUTNIK"
                className="max-w-fit"
            />
            <UploadView />
            <InspectView />
        </div>
    )
}
